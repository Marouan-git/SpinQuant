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