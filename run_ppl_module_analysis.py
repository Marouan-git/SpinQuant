import datetime
from logging import Logger

import os
import json

import time

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
    start_time = time.time()
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

    # Define the output file
    model_name = os.path.basename(model_args.input_model)
    output_dir = "sensitivity_results"
    os.makedirs(output_dir, exist_ok=True)
    results_file = os.path.join(output_dir, f"sensitivity_analysis_module_level_{model_name}_w{ptq_args.w_bits}_a{ptq_args.a_bits}.json")

    all_results = []
    total_layers = 32

    # --- Define all modules to be tested ---
    module_types = [
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj"
    ]
    
    modules_to_test = [f"model.layers.{i}.{mtype}" for i in range(total_layers) for mtype in module_types]

    # --- Add a baseline run: W4A16 (all activations in FP) ---
    print("="*80)
    print("Running baseline: W4A16 (all activations in full precision)")
    ptq_args.quantize_only_module = "DUMMY_MODULE_NAME" # Ensure no module is quantized
    
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
    testloader = data_utils.get_wikitext2(
        seed=ptq_args.seed,
        seqlen=2048,
        tokenizer=tokenizer,
        eval_mode=True,
    )

    ppl, avg_time_per_token = eval_utils.evaluator(model, testloader, utils.DEV, ptq_args)
    
    dist.barrier()
    

    print(f"[SUCCESS] Baseline: PPL = {ppl:.2f}, Time = {avg_time_per_token:.4f} ms/token")

    baseline_result = {
        "module_name": "W4A16_baseline",
        "perplexity": ppl,
        "inference_time_ms_per_token": avg_time_per_token
    }
    all_results.append(baseline_result)
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=4)
    print("="*80)

    # --- Main Analysis Loop ---
    for module_name in modules_to_test:
        print(f"\n[INFO] Running analysis for module: {module_name}")

        ptq_args.quantize_only_module = module_name
        
        # Reload the base model to start fresh for each run
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
        testloader = data_utils.get_wikitext2(
            seed=ptq_args.seed,
            seqlen=2048,
            tokenizer=tokenizer,
            eval_mode=True,
        )

        ppl, avg_time_per_token = eval_utils.evaluator(model, testloader, utils.DEV, ptq_args)

        print(f"[SUCCESS] {module_name}: PPL = {ppl:.2f}, Time = {avg_time_per_token:.4f} ms/token")

        all_results.append({
            "module_name": module_name,
            "perplexity": ppl,
            "inference_time_ms_per_token": avg_time_per_token
        })

        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=4)
        
        print("-" * 80)
    
    end_time = time.time()
    total_seconds = end_time - start_time
    hours, rem = divmod(total_seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    formatted_time = f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"
    print(f"Total analysis time: {formatted_time}")

    time_json_file = f"ppl_analysis_module_time_{model_name}.json"
    with open(time_json_file, 'w') as f:
        json.dump({"total_time": formatted_time}, f, indent=4)

    print("Module-level sensitivity analysis complete.")
    print(f"Final results have been saved to: {results_file}")

if __name__ == "__main__":
    main()