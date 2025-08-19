import datetime
from logging import Logger

import os
import json


import re

import torch
import torch.distributed as dist
from transformers import LlamaTokenizerFast
import transformers
from eval_utils.main import ptq_model
from eval_utils.modeling_llama import LlamaForCausalLM
from utils import data_utils, eval_utils, utils, quant_utils
from utils.process_args import process_args_ptq

def main():
    # --- Setup ---
    # We start by parsing arguments just like in ptq.py
    # We'll override the 'exclude_activations_layers' argument in our loop
    dist.init_process_group(backend="nccl", timeout=datetime.timedelta(hours=8))
    model_args, training_args, ptq_args = process_args_ptq()
    local_rank = utils.get_local_rank()

    print("the rank is {}".format(local_rank))
    torch.distributed.barrier()

    config = transformers.AutoConfig.from_pretrained(
        model_args.input_model, token=model_args.access_token
    )
    # Llama v3.2 specific: Spinquant is not compatiable with tie_word_embeddings, clone lm_head from embed_tokens
    process_word_embeddings = False
    if config.tie_word_embeddings:
        config.tie_word_embeddings = False
        process_word_embeddings = True
    dtype = torch.bfloat16 if training_args.bf16 else torch.float16

    # if not ptq_args.mixed_precision_config:
    #     print("ERROR: This script requires a mixed-precision configuration.")
    #     print("Please provide the path to a JSON file using --mixed_precision_config")
    #     return

    # --- Run the Quantization and Evaluation ---
    # print("="*80)
    # print(f"Running mixed-precision evaluation for {model_args.input_model}")
    # print(f"Low Precision: {ptq_args.a_bits}-bit | High Precision: {ptq_args.high_precision_bits}-bit")
    # print(f"Config File: {ptq_args.mixed_precision_config}")
    # print("="*80)

    # 1. Load the base model
        
    model = LlamaForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_args.input_model,
        config=config,
        torch_dtype=dtype,
        token=model_args.access_token,
    )
    if process_word_embeddings:
        model.lm_head.weight.data = model.model.embed_tokens.weight.data.clone()
    model.cuda()
    current_target_device = model.device

    # 2. Apply quantization settings for the current iteration
    model = ptq_model(ptq_args, model, model_args)
    model.seqlen = training_args.model_max_length

    model.to(current_target_device)

    tokenizer = LlamaTokenizerFast.from_pretrained(
        pretrained_model_name_or_path=model_args.input_model,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=True,
        add_eos_token=False,
        add_bos_token=False,
        token=model_args.access_token,
    )

    
    
    # 3. Evaluate the model to get PPL and Inference Time
    # The 'nb_eval_runs' argument controls how many times evaluation is run for timing

    if ptq_args.eval_dataset == "wikitext2":
        testloader = data_utils.get_wikitext2(
            seed=ptq_args.seed,
            seqlen=2048,
            tokenizer=tokenizer,
            eval_mode=True,
        )
    elif ptq_args.eval_dataset == "c4":
        testloader = data_utils.get_c4(
            seed=ptq_args.seed,
            tokenizer=tokenizer,
            eval_mode=True,
        )

    ppl, avg_time_per_token = eval_utils.evaluator(model, testloader, utils.DEV, ptq_args)
    
    dist.barrier()
    
    # --- Print and Save Results ---
    
    print("\n--- Evaluation Results ---")
    print(f"Perplexity (PPL): {ppl:.4f}")
    print(f"Inference Time:   {avg_time_per_token:.4f} ms/token")
    print("--------------------------\n")

    # Save the results to a file
    model_name = os.path.basename(model_args.input_model)
    output_dir = "mixed_precision_results"
    os.makedirs(output_dir, exist_ok=True)
    results_file_name = os.path.basename(ptq_args.mixed_precision_config)
    results_file_path = os.path.join(output_dir, results_file_name)

    if ptq_args.mixed_precision_config:
        with open(ptq_args.mixed_precision_config, 'r') as f:
            config_data = json.load(f)
        if "layers_in_high_precision" in config_data:
            with open(results_file_path, 'w') as f:
                eval_results = {
                    "model_name": model_name,
                    "low_precision_bits": ptq_args.a_bits,
                    "high_precision_bits": ptq_args.high_precision_bits,
                    "perplexity": ppl,
                    "inference_time_ms_per_token": avg_time_per_token,
                    "mixed_precision_config": ptq_args.mixed_precision_config
                }
                json.dump(eval_results, f, indent=4)
        else:
            with open(results_file_path, 'w') as f:
                eval_results = {
                    "model_name": model_name,
                    "perplexity": ppl,
                    "inference_time_ms_per_token": avg_time_per_token,
                    "mixed_precision_config": ptq_args.mixed_precision_config
                }
                json.dump(eval_results, f, indent=4)
  
    print(f"Results saved to: {results_file_path}")

if __name__ == "__main__":
    main()