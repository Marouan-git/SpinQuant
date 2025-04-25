import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from datasets import load_dataset
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
import math

# --- Kurtosis Calculation ---
def calculate_pearson_kurtosis(tensor: torch.Tensor) -> float:
    t = tensor.detach().flatten().float()
    if t.numel() == 0: return float('nan')
    mean = torch.mean(t)
    std = torch.std(t, unbiased=False)
    if std == 0 or torch.isnan(std) or torch.isinf(std): return float('nan')
    kurt = torch.mean(((t - mean) / std) ** 4)
    return kurt.item()

# --- Activation Capturing using Hooks ---
activation_dict = {}
hook_handles = []
def get_activation_hook(name):
    def hook(module, input, output):
        if isinstance(input, tuple) and len(input) > 0:
            act = input[0].detach().cpu()
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
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16 if args.bf16 else torch.float32,
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
    layer_modules = {}
    print("Registering hooks...")
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
        else: print("Model does not have the expected 'model.model.layers' structure."); return
    except Exception as e: print(f"An error occurred during hook registration: {e}"); return

    # --- Forward Pass ---
    if not hook_handles: print("No hooks registered, cannot capture activations.")
    else:
        try:
            calib_data = get_calib_data(args.calibration_dataset, tokenizer, args.nsamples, args.seed, args.seqlen)
        except Exception as e:
            print(f"Error loading calibration data: {e}")
            for handle in hook_handles: handle.remove(); return

        print(f"Running forward pass on {len(calib_data)} samples to capture activations...")
        with torch.no_grad():
            for i, batch_ids in enumerate(tqdm(calib_data, desc="Calibration Forward Pass")):
                try:
                    try: target_device = model.get_input_embeddings().weight.device
                    except AttributeError: target_device = model.device
                    if isinstance(target_device, str): target_device = torch.device(target_device)
                    input_ids = batch_ids.to(target_device)
                    model(input_ids=input_ids)
                except Exception as e: print(f"\nError during forward pass on sample {i}: {e}")
        print("Forward pass completed.")
        print("Removing hooks...")
        for handle in hook_handles: handle.remove()
        hook_handles.clear()

    # --- Calculate & Filter Activation Kurtosis ---
    activation_kurtosis_results = {}
    layers_to_rotate = []
    selection_method = None
    selection_value = None
    print("Calculating activation kurtosis...")
    if not activation_dict: print("No activations were captured.")
    else:
        for layer_idx in sorted(activation_dict.keys()):
            act_list = activation_dict[layer_idx]
            try:
                all_acts = torch.cat(act_list, dim=0)
                kurt = calculate_pearson_kurtosis(all_acts)
                activation_kurtosis_results[layer_idx] = kurt
                # --- Check against threshold to decide wether to rotate or not ---
                # if not np.isnan(kurt) and kurt > args.kurtosis_threshold:
                #     layers_to_rotate.append(layer_idx)
                # ------------------------------------
            except Exception as e:
                 print(f"Error processing activations for layer {layer_idx}: {e}")
                 activation_kurtosis_results[layer_idx] = float('nan')

    valid_results = {idx: val for idx, val in activation_kurtosis_results.items() if not np.isnan(val)}

    if args.kurtosis_percentile is not None:
        selection_method = "percentile"
        selection_value = args.kurtosis_percentile
        print(f"Selecting top {args.kurtosis_percentile}% of layers based on kurtosis.")
        if valid_results and 0 < args.kurtosis_percentile <= 100:
            sorted_layers = sorted(valid_results.items(), key=lambda item: item[1], reverse=True)
            num_to_select = math.ceil(len(sorted_layers) * (args.kurtosis_percentile / 100.0))
            # Ensure num_to_select is at least 1 if percentile > 0 and list not empty
            num_to_select = max(1, num_to_select) if args.kurtosis_percentile > 0 else num_to_select
            layers_to_rotate = [item[0] for item in sorted_layers[:num_to_select]]
        else:
             print("Warning: Percentile is 0, 100+, or no valid kurtosis values. No layers selected by percentile.")
             layers_to_rotate = [] # Ensure it's empty

    elif args.kurtosis_threshold is not None:
        selection_method = "threshold"
        selection_value = args.kurtosis_threshold
        print(f"Selecting layers with kurtosis > {args.kurtosis_threshold}.")
        layers_to_rotate = [idx for idx, kurt in valid_results.items() if kurt > args.kurtosis_threshold]

    # --- Print Results & Save Filtered List ---
    print("\n--- Activation Kurtosis Results (Pearson's) ---")
    if not activation_kurtosis_results: print("No results found.")
    else:
         print(f"{'Layer':<6} | {'Activation Kurtosis':<20} | {'Rotate?':<7}")
         print("-" * (6 + 3 + 20 + 3 + 7))
         for i, kurt_val in activation_kurtosis_results.items():
              kurt_str = f"{kurt_val:.4f}" if not np.isnan(kurt_val) else "nan"
              rotate_flag = "Yes" if i in layers_to_rotate else "No"
              print(f"{i:<6} | {kurt_str:<20} | {rotate_flag:<7}")

    # --- Save the list of layers to rotate ---
    if args.output_layer_list_path:
        print(f"\nSaving indices of layers with kurtosis > {args.kurtosis_threshold} to {args.output_layer_list_path}...")
        try:
            output_data = {
                "model_path": args.model_path,
                "selection_method": selection_method, # Record method
                "selection_value": selection_value,   # Record value
                "layers_to_rotate": sorted(layers_to_rotate) # Save sorted list
            }
            with open(args.output_layer_list_path, 'w') as f:
                json.dump(output_data, f, indent=4) # Save as JSON for structure
            print("Layer list saved successfully.")
        except Exception as e:
            print(f"Error saving layer list to file: {e}")
    # ---------------------------------------------

    # --- Plotting Results ---
    plot_threshold_line = args.kurtosis_threshold is not None # Only plot line if threshold used

    # --- Plotting Results (keep as before, using log scale) ---
    if activation_kurtosis_results and args.plot_output_path:
        print(f"\nGenerating plot and saving to {args.plot_output_path}...")
        try:
            plot_layers = [idx for idx, val in activation_kurtosis_results.items() if not np.isnan(val)]
            plot_values = [max(val, 1e-6) for val in activation_kurtosis_results.values() if not np.isnan(val)]
            if not plot_layers: print("No valid kurtosis values to plot.")
            else:
                # plt.figure(figsize=(15, 6))
                # colors = ['red' if layer in layers_to_rotate else 'skyblue' for layer in plot_layers] # Color selected bars
                # plt.bar(plot_layers, plot_values, color=colors)
                # plt.yscale('log') # Use log scale
                # plt.axhline(y=args.kurtosis_threshold, color='gray', linestyle='--', label=f'Threshold ({args.kurtosis_threshold})') # Add threshold line
                # plt.xlabel("Layer Index")
                # plt.ylabel("Activation Kurtosis (Pearson's) - Log Scale")
                # plt.title(f"Input Activation Kurtosis for mlp.down_proj Layers\nModel: {args.model_path}")
                # plt.xticks(ticks=plot_layers, fontsize=8)
                # all_layer_indices = sorted(list(activation_kurtosis_results.keys()))
                # if all_layer_indices: plt.xlim(min(all_layer_indices)-0.5, max(all_layer_indices)+0.5)
                # plt.grid(axis='y', linestyle='--', alpha=0.7)
                # plt.legend() # Show threshold label
                # plt.tight_layout()
                # plt.savefig(args.plot_output_path)
                # print("Plot saved successfully.")
                plt.figure(figsize=(15, 6))
                colors = ['red' if layer in layers_to_rotate else 'skyblue' for layer in plot_layers]
                plt.bar(plot_layers, plot_values, color=colors)
                plt.yscale('log')
                if plot_threshold_line: # Only plot threshold line if threshold was used
                    plt.axhline(y=args.kurtosis_threshold, color='gray', linestyle='--', label=f'Threshold ({args.kurtosis_threshold})')
                    plt.legend() # Show legend only if threshold line is present

                plt.xlabel("Layer Index"); plt.ylabel("Activation Kurtosis (Pearson's) - Log Scale")
                title = f"Input Activation Kurtosis for mlp.down_proj Layers\nModel: {args.model_path}"
                if selection_method: title += f"\nSelection: {selection_method}={selection_value}" # Add selection info
                plt.title(title)
                plt.xticks(ticks=plot_layers, fontsize=8)
                all_layer_indices = sorted(list(activation_kurtosis_results.keys()))
                if all_layer_indices: plt.xlim(min(all_layer_indices)-0.5, max(all_layer_indices)+0.5)
                plt.grid(axis='y', linestyle='--', alpha=0.7); plt.tight_layout()
                plt.savefig(args.plot_output_path); print("Plot saved successfully.")
        except Exception as e: print(f"Error generating plot: {e}")

    print("\nCalculation finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate activation kurtosis and identify layers exceeding a threshold.")
    # Model Args
    parser.add_argument("--model_path", type=str, required=True, help="Path to the pretrained model directory or model name.")
    parser.add_argument("--access_token", type=str, default=None, help="Hugging Face access token.")
    parser.add_argument("--bf16", action="store_true", help="Use bfloat16 dtype for model loading.")

    # Data Args
    parser.add_argument("--calibration_dataset", type=str, default="wikitext2", help="Dataset for calibration.")
    parser.add_argument("--nsamples", type=int, default=32, help="Number of calibration samples.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for calibration data sampling.")
    parser.add_argument("--seqlen", type=int, default=1024, help="Sequence length for calibration.")

    # Runtime Args
    parser.add_argument("--device", type=str, default="auto", help="Device: 'cpu', 'cuda', 'cuda:0', 'auto'.")

    # Analysis Args
    #parser.add_argument("--kurtosis_threshold", type=float, required=True, help="Kurtosis value above which a layer is selected for rotation.")
    parser.add_argument("--output_layer_list_path", type=str, default="layers_to_rotate.json", help="Path to save the JSON list of layer indices to rotate.")
    parser.add_argument("--plot_output_path", type=str, default="activation_kurtosis.png", help="Path to save the output plot image.")

    group = parser.add_mutually_exclusive_group(required=True) # One MUST be provided
    group.add_argument("--kurtosis_threshold", type=float, help="Kurtosis value above which a layer is selected.")
    group.add_argument("--kurtosis_percentile", type=float, help="Top X percent of layers with highest kurtosis to select (e.g., 10 for top 10%). Range (0, 100].")

    args = parser.parse_args()

    if args.kurtosis_percentile is not None and not (0 < args.kurtosis_percentile <= 100):
         parser.error("--kurtosis_percentile must be between 0 (exclusive) and 100 (inclusive).")

    # --- Device validation (keep as before) ---
    if "cuda" in args.device or args.device == "auto":
        if not torch.cuda.is_available(): print(f"Warning: CUDA device '{args.device}' requested but unavailable. Using CPU."); args.device = "cpu"
        elif ":" in args.device :
             try: torch.cuda.device(args.device)
             except Exception as e: print(f"Warning: Invalid CUDA device '{args.device}'. Using default. Error: {e}"); args.device = "cuda"

    activation_dict.clear(); hook_handles.clear()
    main(args)