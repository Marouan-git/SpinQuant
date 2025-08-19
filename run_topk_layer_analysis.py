import datetime
import os
import json
import torch
import torch.nn.functional as F
import torch.distributed as dist
from transformers import LlamaTokenizerFast
import transformers
from tqdm import tqdm

from eval_utils.main import ptq_model
from eval_utils.modeling_llama import LlamaForCausalLM
from utils import data_utils, eval_utils, utils
from utils.process_args import process_args_ptq

# --- Helper Functions for the New Metrics ---

def get_top_k(logits, k):
    """Gets the top k indices and corresponding probabilities."""
    top_k_values, top_k_indices = torch.topk(logits, k, dim=-1)
    top_k_probs = F.softmax(top_k_values, dim=-1)
    return top_k_indices, top_k_probs

def jaccard_distance(set1, set2):
    """Calculates the Jaccard distance between two sets of indices."""
    intersection = len(set1.intersection(set2))
    union = len(set1) + len(set2) - intersection
    return 1.0 - (intersection / union) if union > 0 else 0.0

def compute_jsd_on_probs(p_probs, q_probs, base=2.0, eps=1e-6):
    """Computes JSD on already-softmaxed probability vectors."""
    m = 0.5 * (p_probs + q_probs)
    log_base = torch.log(torch.tensor(base, device=p_probs.device))
    kl_p_m = (p_probs * (torch.log(p_probs / (m + eps)) / log_base)).sum(dim=-1)
    kl_q_m = (q_probs * (torch.log(q_probs / (m + eps)) / log_base)).sum(dim=-1)
    jsd = 0.5 * (kl_p_m + kl_q_m)
    return jsd.mean().item()

def normalize_scores(scores_dict):
    """Normalizes a dictionary of scores to the [0, 1] range."""
    scores = list(scores_dict.values())
    min_score, max_score = min(scores), max(scores)
    if max_score == min_score:
        return {k: 0.0 for k in scores_dict}
    return {k: (v - min_score) / (max_score - min_score) for k, v in scores_dict.items()}

# --- Main Analysis Script ---

def main():
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

    model_name = os.path.basename(model_args.input_model)
    output_dir = "topk_sensitivity_results"
    os.makedirs(output_dir, exist_ok=True)
    results_file = os.path.join(output_dir, f"topk_analysis_{model_name}_w{ptq_args.w_bits}_a{ptq_args.a_bits}_k{ptq_args.top_k}.json")

    final_results = {}
    total_layers = 32
    k = ptq_args.top_k
    beta = ptq_args.beta

    print("="*80)
    print(f"Starting Top-{k} Hybrid Sensitivity Analysis for {model_name}")
    print(f"Using {ptq_args.nsamples} samples. Beta = {beta}")
    print(f"Results will be saved to: {results_file}")
    print("="*80)

    # --- Prepare Data and Reference Logits ---
    tokenizer = LlamaTokenizerFast.from_pretrained(
        pretrained_model_name_or_path=model_args.input_model, model_max_length=training_args.model_max_length, token=model_args.access_token
    )
    testloader = data_utils.get_wikitext2(nsamples=ptq_args.nsamples, seed=ptq_args.seed, seqlen=2048, tokenizer=tokenizer, eval_mode=True)

    print("\n[INFO] Getting reference logits from W4A16KV16 model...")
    ptq_args.exclude_activations_layers = list(range(total_layers))

    model_ref = LlamaForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_args.input_model, config=config, torch_dtype=dtype, token=model_args.access_token
    )
    if process_word_embeddings:
            model_ref.lm_head.weight.data = model_ref.model.embed_tokens.weight.data.clone()
    model_ref.cuda()
    model_ref = ptq_model(ptq_args, model_ref, model_args)
    model_ref.seqlen = training_args.model_max_length
    reference_logits_list = eval_utils.get_logits_for_analysis(model_ref, testloader, utils.DEV, ptq_args)
    print(f"[SUCCESS] Captured {len(reference_logits_list)} reference logit batches.")
    print("-" * 80)
    del model_ref

    # --- Main Analysis Loop ---
    for i in range(total_layers):
        print(f"\n[INFO] Running analysis for layer {i} (quantizing ONLY this layer's activations)")

        layers_to_exclude = [j for j in range(total_layers) if j != i]
        ptq_args.exclude_activations_layers = layers_to_exclude
        
        model_quant = LlamaForCausalLM.from_pretrained(
            pretrained_model_name_or_path=model_args.input_model, config=config, torch_dtype=dtype, token=model_args.access_token
        )
        if process_word_embeddings:
            model_quant.lm_head.weight.data = model_quant.model.embed_tokens.weight.data.clone()
        model_quant.cuda()
        model_quant = ptq_model(ptq_args, model_quant, model_args)
        model_quant.seqlen = training_args.model_max_length
        quantized_logits_list = eval_utils.get_logits_for_analysis(model_quant, testloader, utils.DEV, ptq_args)

        confidence_shifts = []
        rank_instabilities = []

        for ref_logits, quant_logits in zip(reference_logits_list, quantized_logits_list):
            ref_logits, quant_logits = ref_logits.to(utils.DEV), quant_logits.to(utils.DEV)
            
            # Get top-k indices and probs from the reference model
            ref_top_k_indices, _ = get_top_k(ref_logits, k)

            # Get top-k indices from the quantized model for Jaccard
            quant_top_k_indices, _ = get_top_k(quant_logits, k)

            # --- Calculate Rank Instability (Jaccard) ---
            # The logits have shape (batch_size, seq_len, vocab_size).
            # We must iterate through both batch and sequence dimensions.
            print(f"top k indices shape: {ref_top_k_indices.shape}")
            for b in range(ref_top_k_indices.shape[0]): # Batch dimension
                for s in range(ref_top_k_indices.shape[1]): # Sequence dimension
                    # Select the 1D tensor of top-k indices for this specific token prediction
                    ref_set = set(ref_top_k_indices[b, s].cpu().tolist())
                    quant_set = set(quant_top_k_indices[b, s].cpu().tolist())
                    rank_instabilities.append(jaccard_distance(ref_set, quant_set))
            
            # --- Calculate Confidence Shift (JSD) ---
            # Get the probabilities of the REFERENCE top-k tokens from BOTH models
            ref_top_k_probs_from_ref = F.softmax(torch.gather(ref_logits, -1, ref_top_k_indices), dim=-1)
            ref_top_k_probs_from_quant = F.softmax(torch.gather(quant_logits, -1, ref_top_k_indices), dim=-1)
            
            confidence_shifts.append(compute_jsd_on_probs(ref_top_k_probs_from_ref, ref_top_k_probs_from_quant))

        avg_confidence_shift = sum(confidence_shifts) / len(confidence_shifts)
        avg_rank_instability = sum(rank_instabilities) / len(rank_instabilities)
        
        dist.barrier()
        print(f"[SUCCESS] Layer {i}: Avg Confidence Shift (JSD) = {avg_confidence_shift:.6f}, Avg Rank Instability (Jaccard) = {avg_rank_instability:.6f}")

        final_results[i] = {
            "confidence_shift": avg_confidence_shift,
            "rank_instability": avg_rank_instability
        }
        del model_quant

    # --- Post-processing: Normalization and Hybrid Score ---
    print("\n" + "="*80)
    print("All layers processed. Normalizing scores and calculating hybrid score...")
    
    confidence_scores = {layer: data['confidence_shift'] for layer, data in final_results.items()}
    instability_scores = {layer: data['rank_instability'] for layer, data in final_results.items()}

    norm_confidence = normalize_scores(confidence_scores)
    norm_instability = normalize_scores(instability_scores)

    for i in range(total_layers):
        final_results[i]['normalized_confidence_shift'] = norm_confidence[i]
        final_results[i]['normalized_rank_instability'] = norm_instability[i]
        final_results[i]['hybrid_score'] = (beta * norm_confidence[i]) + ((1 - beta) * norm_instability[i])

    # Convert to a list of dicts for nice JSON formatting
    output_list = [{"layer_id": k, **v} for k, v in sorted(final_results.items())]

    with open(results_file, 'w') as f:
        json.dump(output_list, f, indent=4)
        
    print("Top-k hybrid sensitivity analysis complete.")
    print(f"Final results have been saved to: {results_file}")

if __name__ == "__main__":
    main()