import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from datasets import load_dataset
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
import math
import os

from utils import quant_utils



# --- Quantization Error Calculation ---
def calculate_quantization_error(tensor: torch.Tensor, quantizer: quant_utils.ActQuantizer, device) -> float:
    """
    Quantizes and dequantizes a tensor, then calculates MSE loss.

    Args:
        tensor: The original input tensor (on CPU).
        quantizer: Configured ActQuantizer instance.
        device: The device to perform quantization on.

    Returns:
        The MSE quantization error as a float. Returns NaN on error.
    """
    if tensor.numel() == 0:
        return float('nan')

    try:
        # Move original tensor to compute device and ensure float32
        orig_tensor_dev = tensor.to(device).float()

        # Find quantization parameters (scale/zero-point)
        quantizer.find_params(orig_tensor_dev)

        # Quantize and dequantize
        q_tensor_dev = quantizer(orig_tensor_dev)

        # Calculate MSE loss
        loss = torch.mean((orig_tensor_dev - q_tensor_dev)**2)

        # Free quantizer state if needed
        if hasattr(quantizer, 'free'):
            quantizer.free()

        return loss.item()

    except Exception as e:
        print(f"Error during quantization/error calculation: {e}")
        # Ensure quantizer state is freed on error too
        if hasattr(quantizer, 'free'):
            quantizer.free()
        return float('nan')


# --- Activation Capturing using Hooks ---
# (Keep activation_dict, hook_handles, get_activation_hook as before)
activation_dict = {}
hook_handles = []
def get_activation_hook(name):
    def hook(module, input, output):
        if isinstance(input, tuple) and len(input) > 0:
            act = input[0].detach().cpu() # Store FP32/BF16 activation on CPU
            if name not in activation_dict: activation_dict[name] = []
            activation_dict[name].append(act)
        else: print(f"Warning: Unexpected input type to hook for {name}: {type(input)}")
    return hook

# --- Data Loading ---
def get_calib_data(name, tokenizer, nsamples, seed, seqlen=2048):
    print(f"Loading calibration data ({name})...")
    if name == "wikitext2":
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train", trust_remote_code=True)
        text = "\n\n".join(dataset["text"])
    elif name == "c4":
        print("Warning: C4 loading not fully implemented here, using wikitext2 as fallback.")
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train", trust_remote_code=True)
        text = "\n\n".join(dataset["text"])
    else: raise ValueError(f"Unknown calibration dataset {name}")

    print("Tokenizing data...")
    enc = tokenizer(text, return_tensors="pt", truncation=False)
    seq = enc.input_ids[0]

    print(f"Using sequence length: {seqlen}, number of samples: {nsamples}")
    nsamples = min(nsamples, seq.numel() // seqlen)
    if nsamples == 0: raise ValueError(f"Dataset too short for seqlen {seqlen}.")

    np.random.seed(seed)
    torch.manual_seed(seed)
    samples = []
    processed_indices = set()
    attempts = 0
    max_attempts = nsamples * 5
    while len(samples) < nsamples and attempts < max_attempts:
        start_index = np.random.randint(0, seq.numel() - seqlen)
        if start_index not in processed_indices:
           end_index = start_index + seqlen
           sample_ids = seq[start_index:end_index].unsqueeze(0)
           samples.append(sample_ids)
           processed_indices.add(start_index)
        attempts += 1
    if len(samples) < nsamples: print(f"Warning: Could only generate {len(samples)} unique samples.")
    print(f"Loaded {len(samples)} calibration samples.")
    return samples

# --- Main Function ---
def main(args):
    # --- Model & Tokenizer Loading ---
    print(f"Loading model from: {args.model_path}")
    model_dtype = torch.bfloat16 if args.bf16 else torch.float32 # Use float32 as default if not bf16
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path, torch_dtype=model_dtype,
            low_cpu_mem_usage=True, device_map=args.device, token=args.access_token, trust_remote_code=True
        )
        model.eval(); print("Model loaded successfully.")
    except Exception as e: print(f"Error loading model: {e}"); return
    print(f"Loading tokenizer for: {args.model_path}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
             args.model_path, model_max_length=args.seqlen, padding_side="right",
             use_fast=True, add_eos_token=False, add_bos_token=False, token=args.access_token, trust_remote_code=True
        )
        print("Tokenizer loaded successfully.")
    except Exception as e: print(f"Error loading tokenizer: {e}"); return

    # --- Hook Registration ---
    layer_modules = {} # Keep track
    print("Registering hooks for down_proj inputs...")
    try:
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            num_layers = len(model.model.layers)
            print(f"Found {num_layers} layers.")
            for i, layer in enumerate(model.model.layers):
                if hasattr(layer, 'mlp') and hasattr(layer.mlp, 'down_proj'):
                    down_proj_layer = layer.mlp.down_proj
                    layer_modules[i] = down_proj_layer
                    handle = down_proj_layer.register_forward_hook(get_activation_hook(i))
                    hook_handles.append(handle)
                else: print(f"  Layer {i}: No 'mlp.down_proj' structure found.")
        else: print("Model does not have expected structure."); return
    except Exception as e: print(f"An error occurred during hook registration: {e}"); return

    # --- Forward Pass (Capture Activations) ---
    if not hook_handles: print("No hooks registered."); return
    try: 
        calib_data = get_calib_data(args.calibration_dataset, tokenizer, args.nsamples, args.seed, args.seqlen)
    except Exception as e: 
        print(f"Error loading calib data: {e}")
        [h.remove() for h in hook_handles]; return
    print(f"Running forward pass on {len(calib_data)} samples...");
    with torch.no_grad():
        for i, batch_ids in enumerate(tqdm(calib_data, desc="Calibration Forward Pass")):
            try:
                try: target_device = model.get_input_embeddings().weight.device
                except AttributeError: target_device = model.device
                if isinstance(target_device, str): target_device = torch.device(target_device)
                input_ids = batch_ids.to(target_device)
                model(input_ids=input_ids)
            except Exception as e: print(f"\nError during forward pass on sample {i}: {e}")
    print("Forward pass completed."); print("Removing hooks..."); [h.remove() for h in hook_handles]; hook_handles.clear()

    # --- Calculate Activation Quantization Error ---
    activation_qerror_results = {}
    layers_with_high_error = []
    selection_method = None
    selection_value = None
    print(f"Calculating activation quantization error (A{args.a_bits} G{args.a_groupsize} Sym={not args.a_asym})...")

    if not activation_dict: print("No activations were captured.")
    else:
        # --- Create and configure the quantizer once ---
        act_quantizer = quant_utils.ActQuantizer()
        act_quantizer.configure(
            bits=args.a_bits,
            groupsize=args.a_groupsize, # Using -1 for token-wise, otherwise channel-wise if > 0
            sym=not args.a_asym,
            clip_ratio=1.0 # Default clipping, add arg if needed
        )
        # --------------------------------------------

        compute_device = torch.device(args.device if args.device != 'auto' and torch.cuda.is_available() else 'cuda:0' if torch.cuda.is_available() else 'cpu')
        print(f"Using device for quantization calculation: {compute_device}")

        for layer_idx in tqdm(sorted(activation_dict.keys()), desc="Analyzing Layers"):
            act_list = activation_dict[layer_idx]
            try:
                # Concatenate all activations for this layer
                all_acts = torch.cat(act_list, dim=0) # Shape: (nsamples, seqlen, hidden_dim)
                # Decide on error aggregation: calculate error per sample/token and average?
                # Or calculate on the whole tensor? Let's calculate on the whole tensor for simplicity.
                # Reshape for token-wise or group-wise quantization if needed by find_params/quantizer
                # Assuming token-wise (groupsize=-1) for now based on ActQuantizer typical use

                mse_error = calculate_quantization_error(all_acts, act_quantizer, compute_device)
                activation_qerror_results[layer_idx] = mse_error

            except Exception as e:
                 print(f"Error processing activations for layer {layer_idx}: {e}")
                 activation_qerror_results[layer_idx] = float('nan')

    # --- Select Layers with HIGHEST Error ---
    valid_results = {idx: val for idx, val in activation_qerror_results.items() if not np.isnan(val)}

    if args.error_percentile is not None:
        selection_method = "percentile_highest_error"
        selection_value = args.error_percentile
        print(f"Selecting {args.error_percentile}% of layers with the HIGHEST quantization error.")
        if valid_results and 0 < args.error_percentile <= 100:
            # Sort by error ascending (highest error first)
            sorted_layers = sorted(valid_results.items(), key=lambda item: item[1], reverse=True)
            num_to_select = math.ceil(len(sorted_layers) * (args.error_percentile / 100.0))
            num_to_select = max(1, num_to_select) if args.error_percentile > 0 else num_to_select
            layers_with_high_error = [item[0] for item in sorted_layers[:num_to_select]]
        else:
             print("Warning: Percentile is 0, 100+, or no valid error values. No layers selected.")
             layers_with_high_error = []
    # Add threshold logic if needed later (selecting BELOW threshold)
    # elif args.error_threshold is not None: ...

    # --- Print Results & Save Filtered List ---
    print("\n--- Activation Quantization Error (MSE) ---")
    if not activation_qerror_results: print("No results found.")
    else:
         print(f"{'Layer':<6} | {'Quantization Error (MSE)':<26} | {'Selected (High Err)?':<18}")
         print("-" * (6 + 3 + 26 + 3 + 18))
         for i in sorted(activation_qerror_results.keys()):
              err_val = activation_qerror_results.get(i, float('nan'))
              # Use scientific notation for potentially small MSE values
              err_str = f"{err_val:.4e}" if not np.isnan(err_val) else "nan"
              select_flag = "Yes" if i in layers_with_high_error else "No"
              print(f"{i:<6} | {err_str:<26} | {select_flag:<18}")

    # --- Save the list of layers selected ---
    if args.output_layer_list_path:
        print(f"\nSaving indices of layers with highest {args.error_percentile}% error to {args.output_layer_list_path}...")
        try:
            output_data = {
                "model_path": args.model_path,
                "quant_config": f"A{args.a_bits}G{args.a_groupsize}Sym{not args.a_asym}",
                "selection_method": selection_method,
                "selection_value": selection_value,
                "layers_to_rotate": sorted(layers_with_high_error)
            }
            with open(args.output_layer_list_path, 'w') as f:
                json.dump(output_data, f, indent=4)
            print("Layer list saved successfully.")
        except Exception as e:
            print(f"Error saving layer list to file: {e}")

    # --- Plotting Results ---
    if activation_qerror_results and args.plot_output_path:
        print(f"\nGenerating plot and saving to {args.plot_output_path}...")
        try:
            plot_layers = [idx for idx, val in activation_qerror_results.items() if not np.isnan(val)]
            plot_values = [max(val, 1e-9) for idx, val in activation_qerror_results.items() if not np.isnan(val)] # Avoid log(0)
            if not plot_layers: print("No valid error values to plot.")
            else:
                plt.figure(figsize=(15, 6))
                colors = ['red' if layer in layers_with_high_error else 'skyblue' for layer in plot_layers]
                plt.bar(plot_layers, plot_values, color=colors)
                plt.yscale('log') # Use log scale for error as it can vary widely
                plt.xlabel("Layer Index")
                plt.ylabel("Activation Quantization Error (MSE) - Log Scale")
                title = f"Input Activation Quant Error (A{args.a_bits}) for mlp.down_proj Layers\nModel: {args.model_path}"
                if selection_method: title += f"\nSelection: {selection_method}={selection_value}% (Highest Error)"
                plt.title(title)
                plt.xticks(ticks=plot_layers, rotation=90, fontsize=8)
                all_layer_indices = sorted(list(activation_qerror_results.keys()))
                if all_layer_indices: plt.xlim(min(all_layer_indices)-0.5, max(all_layer_indices)+0.5)
                plt.grid(axis='y', linestyle='--', alpha=0.7); plt.tight_layout()
                plt.savefig(args.plot_output_path); print("Plot saved successfully.")
        except Exception as e: print(f"Error generating plot: {e}")

    print("\nCalculation finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate activation quantization error and identify layers with highest error.")
    # --- Model Args ---
    parser.add_argument("--model_path", type=str, required=True, help="Path to model.")
    parser.add_argument("--access_token", type=str, default=None, help="HF access token.")
    parser.add_argument("--bf16", action="store_true", default=True, help="Use bfloat16 model loading.")

    # --- Data Args ---
    parser.add_argument("--calibration_dataset", type=str, default="wikitext2", help="Dataset name.")
    parser.add_argument("--nsamples", type=int, default=32, help="# calibration samples.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--seqlen", type=int, default=2048, help="Sequence length.")

    # --- Quantization Config Args ---
    parser.add_argument("--a_bits", type=int, default=4, help="Bits for activation quantization.")
    parser.add_argument("--a_groupsize", type=int, default=-1, help="Group size for activation quant (-1 for token-wise).")
    parser.add_argument("--a_asym", action='store_true', default=True, help="Use asymmetric activation quantization.")
    # parser.add_argument("--a_clip_ratio", type=float, default=1.0, help="Activation clipping ratio.") # Add if needed

    # --- Runtime Args ---
    parser.add_argument("--device", type=str, default="auto", help="Device: 'cpu', 'cuda', 'auto'.")

    # --- Analysis & Output Args ---
    parser.add_argument("--output_layer_list_path", type=str, default="high_error_layers.json", help="Path to save JSON list.")
    parser.add_argument("--plot_output_path", type=str, default="activation_qerror.png", help="Path to save plot.")

    # --- Selection Args (Mutually Exclusive) ---
    group = parser.add_mutually_exclusive_group(required=True) # Require one selection method
    # group.add_argument("--error_threshold", type=float, help="Select layers with error BELOW this threshold.") # Add if needed
    group.add_argument("--error_percentile", type=float, help="Select layers in the Highest X percent of quantization error (e.g., 10 for highest 10%). Range (0, 100].")

    args = parser.parse_args()

    # --- Percentile argument validation ---
    if args.error_percentile is not None and not (0 < args.error_percentile <= 100):
         parser.error("--error_percentile must be between 0 (exclusive) and 100 (inclusive).")

    # --- Device validation ---
    if "cuda" in args.device or args.device == "auto":
        if not torch.cuda.is_available(): print(f"Warning: CUDA device '{args.device}' unavailable. Using CPU."); args.device = "cpu"
        elif ":" in args.device :
             try: torch.cuda.device(args.device)
             except Exception as e: print(f"Warning: Invalid CUDA device '{args.device}'. Using default. Error: {e}"); args.device = "cuda"

    activation_dict.clear(); hook_handles.clear() # Reset globals
    main(args)