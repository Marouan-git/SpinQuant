# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

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

import lm_eval
from lm_eval.models.huggingface import HFLM
from lm_eval.utils import setup_logging
from lm_eval.utils import handle_non_serializable

from lm_analyser.analysis.analyser import LLMAnalyser


setup_logging("DEBUG")

log: Logger = utils.get_logger("spinquant")

# --- Define HFLM Wrapper ---
class QuantizedLMWrapper(HFLM):
    def __init__(self, model_obj, tokenizer_obj, device="cuda", batch_size=1):
        """
        Passes an already initialized model and tokenizer to HFLM.
        :param model_obj: Your initialized and quantized Hugging Face model.
        :param tokenizer_obj: Your initialized Hugging Face tokenizer.
        :param model_id: A string identifier for your model.
        :param device: The device string (e.g., "cuda", "cpu").
        :param batch_size: The batch size for evaluation.
        """
        super().__init__(
            pretrained=model_obj,
            backend="causal",
            device=device,
            batch_size=batch_size,
            tokenizer=tokenizer_obj,
            trust_remote_code=True,
        )

# --- End HFLM Wrapper ---


def train() -> None:
    dist.init_process_group(backend="nccl", timeout=datetime.timedelta(hours=8))
    model_args, training_args, ptq_args = process_args_ptq()
    local_rank = utils.get_local_rank()

    log.info("the rank is {}".format(local_rank))
    torch.distributed.barrier()

    start_time = datetime.datetime.now()

    config = transformers.AutoConfig.from_pretrained(
        model_args.input_model, token=model_args.access_token
    )
    # Llama v3.2 specific: Spinquant is not compatiable with tie_word_embeddings, clone lm_head from embed_tokens
    process_word_embeddings = False
    if config.tie_word_embeddings:
        config.tie_word_embeddings = False
        process_word_embeddings = True
    dtype = torch.bfloat16 if training_args.bf16 else torch.float16
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

    model = ptq_model(ptq_args, model, model_args)
    model.seqlen = training_args.model_max_length

    model.to(current_target_device)

    task_manager = lm_eval.tasks.TaskManager()


    if local_rank == 0:
        log.info("Model PTQ completed {}".format(model))
        log.info("Start to load tokenizer...")
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

    if ptq_args.run_analysis:
        analyzer = LLMAnalyser(model, tokenizer)

        print("--- Running Magnitude Analysis on Quantized Model ---")
        analyzer.run_activation_magnitude_analysis(
            calib_dataset="wikitext2",
            num_samples=128,
            plot=True,
        )

        print("--- Running Magnitude Analysis on Quantized Model with excluded tokens ---")
        analyzer.run_activation_magnitude_analysis(
            calib_dataset="wikitext2",
            num_samples=128,
            plot=True,
            exclude_tokens=['.', '<s>', '\n'],
        )

        print("--- Running Fisher Information Analysis on Quantized Model ---")
        analyzer.run_fisher_information_analysis(
            calib_dataset="wikitext2",
            num_samples=512,
            plot=True,
        )

    # --- Verification Step to Check Activation dtypes ---
    # print("\n" + "="*60)
    # print("VERIFICATION: Checking activation dtypes after quantization setup...")
    # print("="*60)

    # activation_dtypes = {}

    # def robust_dtype_check_hook(name, module, input, output):
    #     """
    #     A more robust hook that checks for a tensor in the output,
    #     even if the output is a tuple.
    #     """
    #     tensor_to_check = None
    #     if isinstance(output, torch.Tensor):
    #         # The output is a simple tensor
    #         tensor_to_check = output
    #     elif isinstance(output, (list, tuple)) and len(output) > 0 and isinstance(output[0], torch.Tensor):
    #         # The output is a tuple/list, let's check the first element
    #         tensor_to_check = output[0]

    #     if tensor_to_check is not None:
    #         activation_dtypes[name] = tensor_to_check.dtype

    # # Find all activation quantizer wrappers
    # verification_qlayers = quant_utils.find_qlayers(model, layers=[quant_utils.ActQuantWrapper])

    # # --- Debugging Print ---
    # print(f"Found {len(verification_qlayers)} activation wrapper layers to check.")
    # if not verification_qlayers:
    #     print("WARNING: No ActQuantWrapper layers found. No dtypes will be captured.")
    # # -----------------------

    # # Register the forward hook
    # for name, module in verification_qlayers.items():
    #     module.register_forward_hook(
    #         lambda m, i, o, n=name: robust_dtype_check_hook(n, m, i, o)
    #     )

    # print("Running one forward pass to capture dtypes...")
    # try:
    #     trainloader = data_utils.get_wikitext2(nsamples=1, seed=ptq_args.seed, model=model_args.input_model, seqlen=2048)
    #     sample_input = next(iter(trainloader))[0].to(next(model.parameters()).device)
    #     with torch.no_grad():
    #         model(sample_input)
    # except Exception as e:
    #     print(f"\nERROR during verification forward pass: {e}")
    #     print("Skipping dtype check.\n")

    # # Print the results
    # if activation_dtypes:
    #     print("\n--- Captured Activation Dtypes ---")
    #     sorted_dtypes = sorted(activation_dtypes.items(), key=lambda item: int(re.search(r'layers\.(\d+)', item[0]).group(1)) if re.search(r'layers\.(\d+)', item[0]) else float('inf'))
    #     for name, dtype in sorted_dtypes:
    #         print(f"{name:<70} | dtype: {dtype}")
    # else:
    #     print("\n--- No activation dtypes were captured. ---")


    # print("="*60)
    # print("Verification complete. Continuing to evaluation...")
    # print("="*60 + "\n")

    # --- End of Verification Step ---
    # print("Complete tokenizer loading...")
    # model.config.use_cache = False

    # print(f"Instantiating LM Eval Harness wrapper with model on {model.device} and tokenizer.")
    # eval_model = QuantizedLMWrapper(
    #     model_obj=model,
    #     tokenizer_obj=tokenizer,
    #     device=model.device.type,
    # )

    # # 2. Define tasks and parameters
    # tasks_to_run = ["winogrande", "hellaswag", "arc_challenge", "arc_easy", "piqa"]#["arc_easy", "arc_challenge", "boolq", "piqa", "social_iqa", "openbookqa", "winogrande", "hellaswag"]
    # num_fewshot = 0

    # log.info(f"Running lm_eval.simple_evaluate on tasks: {tasks_to_run} with {num_fewshot}-shot")

    # nb_evals = ptq_args.nb_eval_runs

    # task_results = {task: {} for task in tasks_to_run}

    # base_seed = 1234

    # if local_rank == 0:

    #     for task in tasks_to_run:
    #         print()
    #         print(f"Running lm_eval.simple_evaluate on task: {task}")
    #         print()
    #         accuracies = []
    #         stderrs = []
    #         for i in range(nb_evals):
    #             ptq_args.seed = base_seed + i
    #             torch.manual_seed(base_seed + i)
    #             model = LlamaForCausalLM.from_pretrained(
    #                 pretrained_model_name_or_path=model_args.input_model,
    #                 config=config,
    #                 torch_dtype=dtype,
    #                 token=model_args.access_token,
    #             )
    #             if process_word_embeddings:
    #                 model.lm_head.weight.data = model.model.embed_tokens.weight.data.clone()
    #             model.cuda()
    #             current_target_device = model.device
    #             model = ptq_model(ptq_args, model, model_args)
    #             model.seqlen = training_args.model_max_length
    #             model.to(current_target_device)
    #             print("Complete tokenizer loading...")
    #             model.config.use_cache = False

    #             print(f"Instantiating LM Eval Harness wrapper with model on {model.device} and tokenizer.")
    #             eval_model = QuantizedLMWrapper(
    #                 model_obj=model,
    #                 tokenizer_obj=tokenizer,
    #                 device=model.device.type,
    #             )
    #             print(f"Running evaluation {i + 1}/{nb_evals} for task: {task}")
    #             print()
    #             results = lm_eval.simple_evaluate(
    #                 model=eval_model,
    #                 tasks=[task],
    #                 num_fewshot=num_fewshot,
    #                 device=model.device.type,  
    #                 task_manager=task_manager,
    #             )
        
    #             # Extract accuracy and stderr from results
    #             if "acc_norm,none" in results["results"][task].keys():
    #                 accuracies.append(results["results"][task]["acc_norm,none"])
    #                 stderrs.append(results["results"][task]["acc_stderr,none"])
    #             else:
    #                 accuracies.append(results["results"][task]["acc,none"])
    #                 stderrs.append(results["results"][task]["acc_stderr,none"])
            
    #         # Compute variance for accuracies
    #         accuracies_var = torch.var(torch.tensor(accuracies))


    #         # Store the acuracies and stderrs for the task
    #         task_results[task]["accuracies"] = accuracies
    #         task_results[task]["stderrs"] = stderrs
    #         # Store the mean and stderr for the task
    #         task_results[task]["mean_accuracy"] = sum(accuracies) / len(accuracies)
    #         task_results[task]["stderr"] = sum(stderrs) / len(stderrs)
    #         # Store the variance for the task
    #         task_results[task]["variance"] = accuracies_var.item()
    #         # Store nb of evals
    #         task_results[task]["nb_evals"] = nb_evals
            
    #         w_bits = getattr(ptq_args, "w_bits", "unknown")
    #         print(f"w_bits: {w_bits}")
            
    #         a_bits = getattr(ptq_args, "a_bits", "unknown")
    #         print(f"a_bits: {a_bits}")
    #         had_config = getattr(ptq_args, "hadamard_online", "no_had")
    #         if had_config:
    #             had_config = "had"
    #         else:
    #             had_config = "no_had"
    #         optimized_rotation = getattr(ptq_args, "optimized_rotation_path", None)
    #         if optimized_rotation:
    #             optimized_rotation = "offline_learned"
    #         else:
    #             optimized_rotation = "offline_hadamard"

    #         results_task_filepath = f"lm_eval_results_{task}_w{w_bits}_a{a_bits}_{had_config}_{optimized_rotation}.json"
    #         try:
    #             with open(results_task_filepath, "w") as f:
    #                 json.dump(task_results[task], f, indent=2, default=handle_non_serializable, ensure_ascii=False)
    #             log.info(f"LM Evaluation Harness results for {task} saved to: {results_task_filepath}")
    #         except Exception as e:
    #             log.error(f"Failed to save LM Evaluation Harness results: {e}")



    #     results_filepath = f"lm_eval_results_{ptq_args.mixed_precision_config}.json"

    #     try:
    #         with open(results_filepath, "w") as f:
    #             json.dump(task_results, f, indent=2, default=handle_non_serializable, ensure_ascii=False)
    #         log.info(f"LM Evaluation Harness results saved to: {results_filepath}")
    #     except Exception as e:
    #         log.error(f"Failed to save LM Evaluation Harness results: {e}")

    if ptq_args.eval_dataset == "c4":
        log.info("Loading C4 dataset for evaluation...")
        testloader = data_utils.get_c4(
            seed=ptq_args.seed,
            seqlen=2048,
            tokenizer=tokenizer,
            eval_mode=True,
        )
        log.info("C4 dataset loaded successfully.")
        dataset_ppl, avg_time_per_token = eval_utils.evaluator(model, testloader, utils.DEV, ptq_args)
        log.info("C4 ppl is: {}".format(dataset_ppl))
        log.info("Average time per token is: {}".format(avg_time_per_token))
    else:
        log.info("Loading Wikitext2 dataset for evaluation...")
        testloader = data_utils.get_wikitext2(
            seed=ptq_args.seed,
            seqlen=2048,
            tokenizer=tokenizer,
            eval_mode=True,
        )
        log.info("Wikitext2 dataset loaded successfully.")
        dataset_ppl, avg_time_per_token = eval_utils.evaluator(model, testloader, utils.DEV, ptq_args)
        log.info("wiki2 ppl is: {}".format(dataset_ppl))
        log.info("Average time per token is: {}".format(avg_time_per_token))

    end_time = datetime.datetime.now()
    log.info("Total time taken for PTQ and evaluation: {}".format(end_time - start_time))
    # time in minutes and seconds
    log.info("Total time taken for PTQ and evaluation: {} minutes and {} seconds".format(
        (end_time - start_time).seconds // 60,
        (end_time - start_time).seconds % 60,
    ))
    dist.barrier()


if __name__ == "__main__":
    train()