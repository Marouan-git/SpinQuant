import datetime
from logging import Logger

import os
import json

import re

import datetime

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
    print("="*80)
    print(f"Running mixed-precision evaluation for {model_args.input_model}")
    print(f"Low Precision: {ptq_args.a_bits}-bit | High Precision: {ptq_args.high_precision_bits}-bit")
    print(f"Config File: {ptq_args.mixed_precision_config}")
    print("="*80)

    start_time = datetime.datetime.now()


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

    task_manager = lm_eval.tasks.TaskManager()

    # 2. Define tasks and parameters
    tasks_to_run = ["winogrande", "hellaswag", "arc_challenge", "arc_easy", "piqa"]#["arc_easy", "arc_challenge", "boolq", "piqa", "social_iqa", "openbookqa", "winogrande", "hellaswag"]
    num_fewshot = 0

    log.info(f"Running lm_eval.simple_evaluate on tasks: {tasks_to_run} with {num_fewshot}-shot")

    nb_evals = ptq_args.nb_eval_runs

    task_results = {task: {} for task in tasks_to_run}

    torch.manual_seed(ptq_args.seed)
    model = LlamaForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_args.input_model,
        config=config,
        torch_dtype=dtype,
        token=model_args.access_token,
    )
    if process_word_embeddings:
        model.lm_head.weight.data = model.model.embed_tokens.weight.data.clone()
    model.cuda()

    total_layers = 32
    #ptq_args.exclude_activations_layers = list(range(total_layers))

    current_target_device = model.device
    model = ptq_model(ptq_args, model, model_args)
    model.seqlen = training_args.model_max_length
    model.to(current_target_device)
    print("Complete tokenizer loading...")
    model.config.use_cache = False

    print(f"Instantiating LM Eval Harness wrapper with model on {model.device} and tokenizer.")
    eval_model = QuantizedLMWrapper(
        model_obj=model,
        tokenizer_obj=tokenizer,
        device=model.device.type,
    )

    if local_rank == 0:

        for task in tasks_to_run:
            print()
            print(f"Running lm_eval.simple_evaluate on task: {task}")
            print()
            accuracies = []
            stderrs = []
            for i in range(nb_evals):
                
                print(f"Running evaluation {i + 1}/{nb_evals} for task: {task}")
                print()
                results = lm_eval.simple_evaluate(
                    model=eval_model,
                    tasks=[task],
                    num_fewshot=num_fewshot,
                    device=model.device.type,  
                    task_manager=task_manager,
                )
        
                # Extract accuracy and stderr from results
                if "acc_norm,none" in results["results"][task].keys():
                    accuracies.append(results["results"][task]["acc_norm,none"])
                    stderrs.append(results["results"][task]["acc_stderr,none"])
                else:
                    accuracies.append(results["results"][task]["acc,none"])
                    stderrs.append(results["results"][task]["acc_stderr,none"])
            
            # Compute variance for accuracies
            accuracies_var = torch.var(torch.tensor(accuracies))


            # Store the acuracies and stderrs for the task
            task_results[task]["accuracies"] = accuracies
            task_results[task]["stderrs"] = stderrs
            # Store the mean and stderr for the task
            task_results[task]["mean_accuracy"] = sum(accuracies) / len(accuracies)
            task_results[task]["stderr"] = sum(stderrs) / len(stderrs)
            # Store the variance for the task
            task_results[task]["variance"] = accuracies_var.item() if accuracies_var.numel() > 0 else 0.0
            # Store nb of evals
            task_results[task]["nb_evals"] = nb_evals
            
            w_bits = getattr(ptq_args, "w_bits", "unknown")
            print(f"w_bits: {w_bits}")
            
            a_bits = getattr(ptq_args, "a_bits", "unknown")
            print(f"a_bits: {a_bits}")
            had_config = getattr(ptq_args, "hadamard_online", "no_had")
            if had_config:
                had_config = "had"
            else:
                had_config = "no_had"
            optimized_rotation = getattr(ptq_args, "optimized_rotation_path", None)
            if optimized_rotation:
                optimized_rotation = "offline_learned"
            else:
                optimized_rotation = "offline_hadamard"

            # results_task_filepath = f"lm_eval_results_{task}_{ptq_args.mixed_precision_config}.json"
            # try:
            #     with open(results_task_filepath, "w") as f:
            #         json.dump(task_results[task], f, indent=2, default=handle_non_serializable, ensure_ascii=False)
            #     log.info(f"LM Evaluation Harness results for {task} saved to: {results_task_filepath}")
            # except Exception as e:
            #     log.error(f"Failed to save LM Evaluation Harness results: {e}")


        mean_accuracy = sum([task_results[task]["mean_accuracy"] for task in tasks_to_run]) / len(tasks_to_run)
        task_results["mean_accuracy"] = mean_accuracy

        #results_filepath = f"lm_eval_results_mixed_precision_module_wise_{os.path.basename(ptq_args.mixed_precision_config)}"
        results_filepath = f"lm_eval_results_mixed_precision_base_accuracy_llama3.2-3b.json"


        try:
            with open(results_filepath, "w") as f:
                json.dump(task_results, f, indent=2, default=handle_non_serializable, ensure_ascii=False)
            log.info(f"LM Evaluation Harness results saved to: {results_filepath}")
        except Exception as e:
            log.error(f"Failed to save LM Evaluation Harness results: {e}")

        end_time = datetime.datetime.now()
        elapsed_time = end_time - start_time
        
        print(f"Total evaluation time: {elapsed_time}")
        # Print in hours, minutes and seconds
        hours, remainder = divmod(elapsed_time.total_seconds(), 3600)
        minutes, seconds = divmod(remainder, 60)
        print(f"Total evaluation time: {int(hours)}h {int(minutes)}m {int(seconds)}s")

    
    dist.barrier()

if __name__ == "__main__":
    main()