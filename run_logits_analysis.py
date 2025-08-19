import datetime
import os
import json
import torch
import torch.nn.functional as F
import torch.distributed as dist
from transformers import LlamaTokenizerFast
import transformers
from tqdm import tqdm
import numpy as np

from eval_utils.main import ptq_model
from eval_utils.modeling_llama import LlamaForCausalLM
from utils import data_utils, eval_utils, utils
from utils.process_args import process_args_ptq

def find_k_for_threshold(probs, threshold):
    """
    Finds the smallest k such that the cumulative probability of the top-k tokens
    is >= threshold.
    """
    # Sort probabilities in descending order
    sorted_probs, _ = torch.sort(probs, descending=True, dim=-1)
    
    # Compute cumulative sum
    cum_probs = torch.cumsum(sorted_probs, dim=-1)
    
    # Find the first index where the cumulative sum exceeds the threshold
    # The result of `(cum_probs >= threshold)` is a boolean tensor.
    # `torch.argmax` on this tensor will return the index of the first `True`.
    # We add 1 because the indices are 0-based.
    k_tensor = torch.argmax((cum_probs >= threshold).int(), dim=-1) + 1
    
    return k_tensor.cpu().tolist()

def main():
    # --- Setup ---
    dist.init_process_group(backend="nccl", timeout=datetime.timedelta(hours=8))
    model_args, training_args, ptq_args = process_args_ptq()
    torch.distributed.barrier()

    config = transformers.AutoConfig.from_pretrained(
        model_args.input_model, token=model_args.access_token
    )
    process_word_embeddings = False
    if config.tie_word_embeddings:
        config.tie_word_embeddings = False
        process_word_embeddings = True
    dtype = torch.bfloat16 if training_args.bf16 else torch.float16

    # --- File Paths and Config ---
    model_name = os.path.basename(model_args.input_model)
    output_dir = "topk_discovery_results"
    os.makedirs(output_dir, exist_ok=True)
    results_file = os.path.join(output_dir, f"k_discovery_{model_name}_w{ptq_args.w_bits}_p{ptq_args.probability_threshold}.json")

    total_layers = 32
    threshold = ptq_args.probability_threshold

    print("="*80)
    print(f"Starting Top-k Discovery Analysis for {model_name}")
    print(f"Using {ptq_args.nsamples} samples and a probability threshold of {threshold}")
    print(f"Results will be saved to: {results_file}")
    print("="*80)

    # --- Prepare Data and Get Logits ---
    tokenizer = LlamaTokenizerFast.from_pretrained(
        pretrained_model_name_or_path=model_args.input_model, model_max_length=training_args.model_max_length, token=model_args.access_token
    )
    testloader = data_utils.get_wikitext2(nsamples=ptq_args.nsamples, seed=ptq_args.seed, seqlen=2048, tokenizer=tokenizer, eval_mode=True)
    
    print("\n[INFO] Getting logits from W4A16KV16 model...")
    # Exclude all layers from activation quantization to get the W4A16 baseline
    ptq_args.exclude_activations_layers = list(range(total_layers))

    model = LlamaForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_args.input_model, config=config, torch_dtype=dtype, token=model_args.access_token
    )
    if process_word_embeddings:
        model.lm_head.weight.data = model.model.embed_tokens.weight.data.clone()
    model.cuda()
    model = ptq_model(ptq_args, model, model_args)
    model.seqlen = training_args.model_max_length

    # Use the efficient evaluator to get logits for all samples
    logits_list = eval_utils.get_logits_for_analysis(model, testloader, utils.DEV, ptq_args)
    print(f"[SUCCESS] Captured {len(logits_list)} logit batches.")
    print("-" * 80)

    # --- Process Logits to Find K Values ---
    all_k_values = []
    print(f"Processing {len(logits_list)} batches to find k values...")
    for logits_batch in tqdm(logits_list, desc="Analyzing Logits"):
        # Logits have shape (batch_size, seq_len, vocab_size)
        # We need to flatten the first two dimensions to process all token predictions
        num_predictions = logits_batch.shape[0] * logits_batch.shape[1]
        flat_logits = logits_batch.view(num_predictions, -1)
        
        # Convert logits to probabilities
        probs = F.softmax(flat_logits, dim=-1)
        
        # Get the list of k values for this batch
        k_values_for_batch = find_k_for_threshold(probs, threshold)
        all_k_values.extend(k_values_for_batch)

    # --- Aggregate and Analyze the Results ---
    print("\n" + "="*80)
    print("Analysis complete. Calculating statistics...")
    
    # Convert to a NumPy array for efficient statistics
    k_array = np.array(all_k_values)
    
    # Calculate statistics
    stats = {
        "mean": np.mean(k_array),
        "median": np.median(k_array),
        "std_dev": np.std(k_array),
        "min": int(np.min(k_array)),
        "max": int(np.max(k_array)),
        "percentile_75": np.percentile(k_array, 75),
        "percentile_90": np.percentile(k_array, 90),
        "percentile_95": np.percentile(k_array, 95),
        "percentile_99": np.percentile(k_array, 99)
    }

    # Generate data for a histogram
    counts, bin_edges = np.histogram(k_array, bins=50) # You can adjust the number of bins

    # Prepare final JSON output
    output_data = {
        "configuration": {
            "model": model_name,
            "quantization": f"W{ptq_args.w_bits}A16",
            "probability_threshold": threshold,
            "num_samples": ptq_args.nsamples,
            "total_predictions_analyzed": len(all_k_values)
        },
        "statistics": stats,
        "histogram_data": {
            "counts": counts.tolist(),
            "bin_edges": bin_edges.tolist()
        }
    }
    
    print("\n--- Top-k Statistics ---")
    for key, value in stats.items():
        print(f"{key.replace('_', ' ').title():<15}: {value:.2f}")
    print("------------------------")
    
    with open(results_file, 'w') as f:
        json.dump(output_data, f, indent=4)
        
    print(f"\nTop-k discovery analysis complete.")
    print(f"Final results have been saved to: {results_file}")

if __name__ == "__main__":
    main()