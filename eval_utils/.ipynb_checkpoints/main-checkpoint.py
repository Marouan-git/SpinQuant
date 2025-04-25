# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# This code is based on QuaRot(https://github.com/spcl/QuaRot/tree/main/quarot).
# Licensed under Apache License 2.0.

import os
import json

import torch
import transformers

from eval_utils import gptq_utils, rotation_utils
from utils import data_utils, fuse_norm_utils, hadamard_utils, quant_utils, utils
from utils.convert_to_executorch import (
    sanitize_checkpoint_from_spinquant,
    write_model_llama,
)


def ptq_model(args, model, model_args=None):
    transformers.set_seed(args.seed)
    model.eval()

    selective_had_layers = None
    if args.selective_had_layers_path:
        print(f"INFO: Attempting to load selective Hadamard layer list from: {args.selective_had_layers_path}")
        if os.path.exists(args.selective_had_layers_path):
            try:
                with open(args.selective_had_layers_path, 'r') as f:
                    data = json.load(f)
                    if "layers_to_rotate" in data and isinstance(data["layers_to_rotate"], list):
                        selective_had_layers = set(data["layers_to_rotate"]) # Use a set for faster lookups
                        print(f"INFO: Loaded {len(selective_had_layers)} layer indices for selective Hadamard.")
                    else:
                        print("Warning: JSON file found, but 'layers_to_rotate' key missing or not a list. Applying Hadamard to all eligible layers.")
            except Exception as e:
                print(f"Warning: Error loading or parsing JSON file '{args.selective_had_layers_path}': {e}. Applying Hadamard to all eligible layers.")
        else:
            print(f"Warning: Selective Hadamard layer file not found at '{args.selective_had_layers_path}'. Applying Hadamard to all eligible layers.")
    # ------------------------------------------------------

    # Rotate the weights
    if args.rotate:
        fuse_norm_utils.fuse_layer_norms(model)
        # Pass hadamard_online flag if using that approach, otherwise handle logic below
        rotation_utils.rotate_model(model, args)
        utils.cleanup_memory(verbos=True)

        quant_utils.add_actquant(model)

    # Determine if *any* online Hadamard logic should be run
        # It should run if global flag is set OR if a selective list was successfully loaded
        run_online_had_setup = getattr(args, 'hadamard_online', False) or (selective_had_layers is not None)

        if run_online_had_setup:
            print("INFO: Online Hadamard setup needed (Global or Selective).")
            qlayers = quant_utils.find_qlayers(model)
            applied_selectively_count = 0
            configured_globally = False

            for name, layer_module in qlayers.items():
                if "down_proj" in name:
                    layer_idx = -1 # Default invalid index
                    try:
                        # Extract layer index from name
                        layer_idx = int(name.split('.')[2])

                        # Determine if Hadamard should be ON for *this* specific layer
                        enable_had_for_layer = False
                        is_global_mode = getattr(args, 'hadamard_online', False) and selective_had_layers is None

                        if is_global_mode:
                             # Global mode (--hadamard_online present, --selective_had_layers_path absent)
                             enable_had_for_layer = True
                             configured_globally = True # Mark that we used global setting
                        elif selective_had_layers is not None and layer_idx in selective_had_layers:
                             # Selective mode (--selective_had_layers_path present)
                             enable_had_for_layer = True

                        # Apply the configuration based on the decision for this layer
                        if enable_had_for_layer:
                            had_K, K = hadamard_utils.get_hadK(model.config.intermediate_size)
                            layer_module.online_full_had = True
                            layer_module.had_K = had_K
                            layer_module.K = K
                            layer_module.fp32_had = args.fp32_had
                            if selective_had_layers is not None:
                                applied_selectively_count += 1
                                # print(f"DEBUG: Enabling online Had for layer {layer_idx}")
                        else:
                            # Ensure Hadamard is explicitly OFF if not enabled for this layer
                            layer_module.online_full_had = False
                            # print(f"DEBUG: Disabling online Had for layer {layer_idx}")

                    except (IndexError, ValueError):
                         print(f"Warning: Could not parse layer index from name '{name}'. Hadamard setting skipped for this layer.")
                         # Ensure it's off if we can't parse index during selective mode
                         if hasattr(layer_module, 'online_full_had'):
                              layer_module.online_full_had = False

            # --- Logging summary ---
            if selective_had_layers is not None:
                 print(f"INFO: Applied online Hadamard selectively to {applied_selectively_count} down_proj layers based on the provided list.")
            elif configured_globally:
                 print(f"INFO: Applied online Hadamard globally to all down_proj layers.")


        else: # Neither global nor selective hadamard requested
             print("INFO: No online Hadamard requested (global or selective). Ensuring it's off.")
             # Ensure it's off for all layers just in case wrappers were added before this check
             qlayers = quant_utils.find_qlayers(model)
             for name, layer_module in qlayers.items():
                 if "down_proj" in name:
                     if hasattr(layer_module, 'online_full_had'): # Check attribute exists
                          layer_module.online_full_had = False

        # # --- Apply Online Hadamard Conditionally ---
        # # Check if online Hadamard is enabled at all (using the flag from previous discussion)
        # # If you didn't add --hadamard_online, this check might just be implicit (always true if args.rotate is true)
        # apply_online_hadamard = getattr(args, 'hadamard_online', False) # Default to False if flag doesn't exist

        # if apply_online_hadamard:
        #     print("INFO: Base condition for online Hadamard met. Applying selectively if specified...")
        #     qlayers = quant_utils.find_qlayers(model)
        #     applied_selectively_count = 0
        #     for name, layer_module in qlayers.items(): # Iterate directly over items
        #         if "down_proj" in name:
        #             try:
        #                 # Extract layer index from name (assuming format like 'model.layers.N.mlp.down_proj')
        #                 layer_idx = int(name.split('.')[2])

        #                 # Determine if this layer should get Hadamard
        #                 apply_to_this_layer = True # Default if no selective list
        #                 if selective_had_layers is not None:
        #                     apply_to_this_layer = (layer_idx in selective_had_layers)

        #                 if apply_to_this_layer:
        #                     # Apply Hadamard setup
        #                     had_K, K = hadamard_utils.get_hadK(model.config.intermediate_size)
        #                     layer_module.online_full_had = True
        #                     layer_module.had_K = had_K
        #                     layer_module.K = K
        #                     layer_module.fp32_had = args.fp32_had
        #                     if selective_had_layers is not None:
        #                         applied_selectively_count += 1
        #                         # print(f"DEBUG: Applying online Had to layer {layer_idx}") # Optional debug print
        #                 else:
        #                    # Ensure Hadamard is off for layers *not* in the list
        #                    layer_module.online_full_had = False
        #                    # print(f"DEBUG: Skipping online Had for layer {layer_idx}") # Optional debug print

        #             except (IndexError, ValueError):
        #                  print(f"Warning: Could not parse layer index from name '{name}'. Cannot apply selective Hadamard.")
        #                  # Fallback: maybe apply if selective_had_layers is None? Or skip? Skipping is safer.
        #                  layer_module.online_full_had = False

        #     if selective_had_layers is not None:
        #          print(f"INFO: Applied online Hadamard selectively to {applied_selectively_count} down_proj layers based on the provided list.")
        # else:
        #      print("INFO: Online Hadamard flag (--hadamard_online) is False or absent. Skipping setup.")
        #      # Ensure it's off for all layers if the main flag is off
        #      qlayers = quant_utils.find_qlayers(model)
        #      for name, layer_module in qlayers.items():
        #          if "down_proj" in name:
        #               layer_module.online_full_had = False


    else: # No rotation at all
        quant_utils.add_actquant(
            model
        )

    """# Rotate the weights
    if args.rotate:
        fuse_norm_utils.fuse_layer_norms(model)
        rotation_utils.rotate_model(model, args)
        utils.cleanup_memory(verbos=True)

        quant_utils.add_actquant(model)  # Add Activation Wrapper to the model
    
        # --- Conditionally set up online Hadamard ---
        if args.hadamard_online: # Check the NEW flag
            print("INFO: Setting up online Hadamard transforms (SpinQuant_had mode).")
            qlayers = quant_utils.find_qlayers(model)
            for name in qlayers:
                if "down_proj" in name:
                    had_K, K = hadamard_utils.get_hadK(model.config.intermediate_size)
                    qlayers[name].online_full_had = True
                    qlayers[name].had_K = had_K
                    qlayers[name].K = K
                    qlayers[name].fp32_had = args.fp32_had
        else:
             print("INFO: Skipping online Hadamard transforms (SpinQuant_no_had mode).")
        
        '''qlayers = quant_utils.find_qlayers(model)
        for name in qlayers:
            if "down_proj" in name:
                had_K, K = hadamard_utils.get_hadK(model.config.intermediate_size)
                qlayers[name].online_full_had = True
                qlayers[name].had_K = had_K
                qlayers[name].K = K
                qlayers[name].fp32_had = args.fp32_had'''
    else:
        quant_utils.add_actquant(
            model
        )  # Add Activation Wrapper to the model as the rest of the code assumes it is present"""

    if args.w_bits < 16:
        save_dict = {}
        if args.load_qmodel_path:  # Load Quantized Rotated Model
            assert args.rotate, "Model should be rotated to load a quantized model!"
            assert (
                not args.save_qmodel_path
            ), "Cannot save a quantized model if it is already loaded!"
            print("Load quantized model from ", args.load_qmodel_path)
            save_dict = torch.load(args.load_qmodel_path)
            model.load_state_dict(save_dict["model"])

        elif not args.w_rtn:  # GPTQ Weight Quantization
            trainloader = data_utils.get_wikitext2(
                nsamples=args.nsamples,
                seed=args.seed,
                model=model_args.input_model,
                seqlen=2048,
                eval_mode=False,
            )
            if args.export_to_et:
                # quantize lm_head and embedding with 8bit per-channel quantization with rtn for executorch
                quantizers = gptq_utils.rtn_fwrd(
                    model,
                    "cuda",
                    args,
                    custom_layers=[model.model.embed_tokens, model.lm_head],
                )
            # quantize other layers with gptq
            quantizers = gptq_utils.gptq_fwrd(model, trainloader, "cuda", args)
            save_dict["w_quantizers"] = quantizers
        else:  # RTN Weight Quantization
            quantizers = gptq_utils.rtn_fwrd(model, "cuda", args)
            save_dict["w_quantizers"] = quantizers

        if args.save_qmodel_path:
            save_dict["model"] = model.state_dict()
            if args.export_to_et:
                save_dict = write_model_llama(
                    model.state_dict(), model.config, num_shards=1
                )[0]  # Export num_shards == 1 for executorch
                save_dict = sanitize_checkpoint_from_spinquant(
                    save_dict, group_size=args.w_groupsize
                )
            torch.save(save_dict, args.save_qmodel_path)

    # Add Input Quantization
    if args.a_bits < 16 or args.v_bits < 16:
        qlayers = quant_utils.find_qlayers(model, layers=[quant_utils.ActQuantWrapper])
        down_proj_groupsize = -1
        if args.a_groupsize > 0:
            down_proj_groupsize = utils.llama_down_proj_groupsize(
                model, args.a_groupsize
            )

        for name in qlayers:
            layer_input_bits = args.a_bits
            layer_groupsize = args.a_groupsize
            layer_a_sym = not (args.a_asym)
            layer_a_clip = args.a_clip_ratio

            num_heads = model.config.num_attention_heads
            model_dim = model.config.hidden_size
            head_dim = model_dim // num_heads

            if "v_proj" in name and args.v_bits < 16:  # Set the v_proj precision
                v_groupsize = head_dim
                qlayers[name].out_quantizer.configure(
                    bits=args.v_bits,
                    groupsize=v_groupsize,
                    sym=not (args.v_asym),
                    clip_ratio=args.v_clip_ratio,
                )

            if "o_proj" in name:
                layer_groupsize = head_dim

            if "lm_head" in name:  # Skip lm_head quantization
                layer_input_bits = 16

            if "down_proj" in name:  # Set the down_proj precision
                if args.int8_down_proj:
                    layer_input_bits = 8
                layer_groupsize = down_proj_groupsize

            qlayers[name].quantizer.configure(
                bits=layer_input_bits,
                groupsize=layer_groupsize,
                sym=layer_a_sym,
                clip_ratio=layer_a_clip,
            )

    if args.k_bits < 16:
        if args.k_pre_rope:
            raise NotImplementedError("Pre-RoPE quantization is not supported yet!")
        else:
            rope_function_name = "apply_rotary_pos_emb"
            layers = model.model.layers
            k_quant_config = {
                "k_bits": args.k_bits,
                "k_groupsize": args.k_groupsize,
                "k_sym": not (args.k_asym),
                "k_clip_ratio": args.k_clip_ratio,
            }
            for layer in layers:
                rotation_utils.add_qk_rotation_wrapper_after_function_call_in_forward(
                    layer.self_attn,
                    rope_function_name,
                    config=model.config,
                    **k_quant_config,
                )

    return model
