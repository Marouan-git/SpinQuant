# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# This code is based on QuaRot(https://github.com/spcl/QuaRot/tree/main/quarot).
# Licensed under Apache License 2.0.

import logging
import os
import json

import torch
import torch.cuda
import time
from tqdm import tqdm

from utils import model_utils


@torch.no_grad()
def evaluator(model, testenc, dev, args):
    model.eval()

    print("INFO: nb of evaluation runs: ", args.nb_eval_runs)
    max_trials = args.nb_eval_runs

    list_total_inference_time = []
    list_time_per_token = []
    list_ppl = []

    list_total_inference_time_perf_count = []
    list_time_per_token_perf_count = []



    for _ in range(max_trials):

        use_cache = model.config.use_cache
        model.config.use_cache = False

        layers = model.model.layers
        model.model.embed_tokens = model.model.embed_tokens.to(dev)

        layers[0] = layers[0].to(dev)

        seq_len = model.seqlen

        # Convert the whole text of evaluation dataset into batches of sequences.
        input_ids = testenc.input_ids  # (1, text_len)
        nsamples = input_ids.numel() // seq_len  # The tail is truncated.
        input_ids = (
            input_ids[:, : nsamples * seq_len].view(nsamples, seq_len).to(dev)
        )  # (nsamples, seqlen)
        
        total_tokens_processed = nsamples * seq_len

        print(f"INFO: Evaluator using seqlen={seq_len}, found {nsamples} samples ({total_tokens_processed} tokens).")

        batch_size = args.bsz
        input_ids = [input_ids[i : i + batch_size] for i in range(0, nsamples, batch_size)]
        nbatches = len(input_ids)

        dtype = next(iter(model.parameters())).dtype
        # The input of the first decoder layer.
        inps = torch.zeros(
            (nbatches, batch_size, model.seqlen, model.config.hidden_size),
            dtype=dtype,
            device=dev,
        )
        inps = [0] * nbatches
        cache = {"i": 0, "attention_mask": None}

        class Catcher(torch.nn.Module):
            def __init__(self, module):
                super().__init__()
                self.module = module

            def forward(self, inp, **kwargs):
                inps[cache["i"]] = inp
                cache["i"] += 1
                cache["attention_mask"] = kwargs["attention_mask"]
                cache["position_ids"] = kwargs["position_ids"]
                raise ValueError

        layers[0] = Catcher(layers[0])

        for i in range(nbatches):
            batch = input_ids[i]
            try:
                model(batch)
            except ValueError:
                pass
        layers[0] = layers[0].module
        layers[0] = layers[0].cpu()

        model.model.embed_tokens = model.model.embed_tokens.cpu()
        position_ids = cache["position_ids"]

        torch.cuda.empty_cache()
        outs = [0] * nbatches
        attention_mask = cache["attention_mask"]

        # --- Timing Initialization ---
        total_layer_processing_time_ms = 0.0
        total_lm_head_time_ms = 0.0
        start_event_layer_loop = torch.cuda.Event(enable_timing=True)
        end_event_layer_loop = torch.cuda.Event(enable_timing=True)
        start_event_head_loop = torch.cuda.Event(enable_timing=True)
        end_event_head_loop = torch.cuda.Event(enable_timing=True)
        # --- End Timing Initialization ---

        torch.cuda.synchronize()
        start_event_layer_loop.record()
        start_cpu_layer = time.perf_counter()

        for i in tqdm(range(len(layers)), desc="(Eval) Layers"):
            layer = layers[i].to(dev)

            # Dump the layer input and output
            if args.capture_layer_io and args.layer_idx == i:
                captured_io = model_utils.capture_layer_io(layer, inps)
                save_path = model_utils.get_layer_io_save_path(args)
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save(captured_io, save_path)
                logging.info(f"Dumped layer input and output to: {save_path}")

            for j in range(nbatches):
                outs[j] = layer(
                    inps[j],
                    attention_mask=attention_mask,
                    #  defined.
                    position_ids=position_ids,
                )[0]
            layers[i] = layer.cpu()
            del layer
            torch.cuda.empty_cache()
            inps, outs = outs, inps
        
        end_event_layer_loop.record()
        torch.cuda.synchronize()
        end_cpu_layer = time.perf_counter()
        total_layer_processing_time_ms_cpu = (end_cpu_layer - start_cpu_layer) * 1000
        print(f"INFO: Layer processing loop finished. Time (CPU): {total_layer_processing_time_ms_cpu:.2f} ms")
        total_layer_processing_time_ms = start_event_layer_loop.elapsed_time(end_event_layer_loop)
        print(f"INFO: Layer processing loop finished. Time: {total_layer_processing_time_ms:.2f} ms")
        # --- End Layer Processing Loop ---

        if model.model.norm is not None:
            model.model.norm = model.model.norm.to(dev)

        model.lm_head = model.lm_head.to(dev)
        nlls = []
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        print("INFO: Starting LM Head and Loss calculation loop...")
        torch.cuda.synchronize()
        start_event_head_loop.record()
        start_cpu_head = time.perf_counter()

        for i in range(nbatches):
            hidden_states = inps[i]
            if model.model.norm is not None:
                hidden_states = model.model.norm(hidden_states)
            lm_logits = model.lm_head(hidden_states)
            shift_logits = lm_logits[:, :-1, :]
            shift_labels = input_ids[i][:, 1:]
            loss = loss_fct(shift_logits.permute(0, 2, 1), shift_labels)
            neg_log_likelihood = loss.float().mean(dim=1)
            nlls.append(neg_log_likelihood)
        
        end_event_head_loop.record()
        torch.cuda.synchronize()
        end_cpu_head = time.perf_counter()
        total_lm_head_processing_time_cpu = (end_cpu_head - start_cpu_head) * 1000
        print(f"INFO: LM Head and Loss calculation loop finished. Time (CPU): {total_lm_head_processing_time_cpu:.2f} ms")
        total_lm_head_time_ms = start_event_head_loop.elapsed_time(end_event_head_loop)
        print(f"INFO: LM Head loop finished. Time: {total_lm_head_time_ms:.2f} ms")
        # --- End LM Head Loop ---

        nlls_tensor = torch.cat(nlls)
        ppl = torch.exp(nlls_tensor.mean())
        model.config.use_cache = use_cache
        logging.info(f"\n WikiText2 PPL: {ppl.item():.3f}")

        # --- Timing Report ---
        total_inference_time_ms = total_layer_processing_time_ms + total_lm_head_time_ms
        total_inference_time_ms_cpu = total_layer_processing_time_ms_cpu + total_lm_head_processing_time_cpu
        if total_tokens_processed > 0:
            time_per_token_ms = total_inference_time_ms / total_tokens_processed
            time_per_token_ms_cpu = total_inference_time_ms_cpu / total_tokens_processed
            print(f"Total Inference Time (Layers + LM Head) (CPU): {total_inference_time_ms_cpu:.2f} ms")
            print(f"Average Inference Time per Token (CPU): {time_per_token_ms_cpu:.4f} ms/token")
            print(f"Total Inference Time (Layers + LM Head): {total_inference_time_ms:.2f} ms")
            print(f"Average Inference Time per Token: {time_per_token_ms:.4f} ms/token")
            list_total_inference_time.append(total_inference_time_ms)
            list_total_inference_time_perf_count.append(total_inference_time_ms_cpu)
            list_time_per_token.append(time_per_token_ms)
            list_time_per_token_perf_count.append(time_per_token_ms_cpu)
            list_ppl.append(ppl.item())
        else:
            print("No tokens processed, cannot calculate time per token.")
        # --- End Timing Report ---
    avg_total_inference_time = sum(list_total_inference_time) / len(list_total_inference_time)
    var_total_inference_time = sum(
        [(x - avg_total_inference_time) ** 2 for x in list_total_inference_time]
    ) / len(list_total_inference_time)
    avg_total_inference_time_perf_count = sum(list_total_inference_time_perf_count) / len(list_total_inference_time_perf_count)
    var_total_inference_time_perf_count = sum(
        [(x - avg_total_inference_time_perf_count) ** 2 for x in list_total_inference_time_perf_count]
    ) / len(list_total_inference_time_perf_count)

    avg_time_per_token = sum(list_time_per_token) / len(list_time_per_token)
    var_time_per_token = sum(
        [(x - avg_time_per_token) ** 2 for x in list_time_per_token]
    ) / len(list_time_per_token)
    avg_time_per_token_perf_count = sum(list_time_per_token_perf_count) / len(list_time_per_token_perf_count)
    var_time_per_token_perf_count = sum(
        [(x - avg_time_per_token_perf_count) ** 2 for x in list_time_per_token_perf_count]
    ) / len(list_time_per_token_perf_count)

    avg_ppl = sum(list_ppl) / len(list_ppl)

    # --- Save Detailed Results to JSON if path is provided ---
    if hasattr(args, 'timing_output_path') and args.timing_output_path:
        print(f"INFO: Saving detailed timing results to {args.timing_output_path}")
        results_to_save = {
            "nb_runs": max_trials,
            "list_ppl": list_ppl,
            "avg_ppl": avg_ppl,
            "cuda_timing_ms": {
                "list_total_time": list_total_inference_time,
                "list_token_time": list_time_per_token,
                "avg_total_time": avg_total_inference_time,
                "var_total_time": var_total_inference_time,
                "avg_token_time": avg_time_per_token,
                "var_token_time": var_time_per_token,
            },
            "cpu_timing_ms": {
                "avg_total_time": avg_total_inference_time_perf_count,
                "var_total_time": var_total_inference_time_perf_count,
                "avg_token_time": avg_time_per_token_perf_count,
                "var_token_time": var_time_per_token_perf_count,
            }
        }
        try:
            with open(args.timing_output_path, 'w') as f:
                json.dump(results_to_save, f, indent=4, default=lambda o: '<not serializable>')
            print(f"INFO: Successfully saved timing results to {args.timing_output_path}")
        except Exception as e:
            print(f"ERROR: Could not save timing results to {args.timing_output_path}: {e}")
    # --- End JSON Saving ---

    print(f"Average Total Inference Time: {avg_total_inference_time:.2f} ms")
    print(f"Average Total Inference Time (CPU): {avg_total_inference_time_perf_count:.2f} ms")
    print(f"Variance Total Inference Time: {var_total_inference_time:.2f} ms")
    print()
    print(f"Average Time per Token: {avg_time_per_token:.4f} ms/token")
    print(f"Average Time per Token (CPU): {avg_time_per_token_perf_count:.4f} ms/token")
    print(f"Variance Time per Token: {var_time_per_token:.4f} ms/token")
    print()
    print(f"Average PPL: {avg_ppl:.3f}")
    return avg_ppl
