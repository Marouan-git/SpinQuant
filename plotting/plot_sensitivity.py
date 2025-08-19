import json
import matplotlib.pyplot as plt
import numpy as np
import re
from collections import defaultdict
import argparse

def plot_sensitivity_results(json_file_path, level='module'):
    """
    Reads sensitivity analysis results from a JSON file and generates plots.
    The type of plot is determined by the 'level' argument.

    Args:
        json_file_path (str): The path to the input JSON file.
        level (str): The granularity of the plot. Can be 'module' or 'layer'.
    """
    # --- 1. Load the data from the JSON file ---
    try:
        with open(json_file_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: The file '{json_file_path}' was not found.")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from the file '{json_file_path}'.")
        return

    # --- 2. Generate plot based on the specified level ---
    if level == 'module':
        plot_module_wise(data)
    elif level == 'layer':
        plot_layer_wise(data)
    else:
        # This case is technically handled by argparse's `choices`, but it's good practice
        print(f"Error: Invalid plot level '{level}'. Please choose 'module' or 'layer'.")

def plot_layer_wise(data):
    """Generates and saves a layer-wise sensitivity plot."""
    print("Generating layer-wise plot...")
    baseline_perplexity = None
    layer_ids = [item['layer_id'] for item in data]
    perplexities = [item['perplexity'] for item in data]
    inference_times = [item['inference_time_ms_per_token'] for item in data]

    for item in data:
        layer_id = item['layer_id']
        if layer_id == "W4A16_baseline":
            baseline_perplexity = item['perplexity']
            continue

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax1 = plt.subplots(1, 1, figsize=(12, 7), sharex=True)

    if baseline_perplexity is not None:
        ax1.axhline(y=baseline_perplexity, color='k', linestyle='--', label=f'Baseline PPL ({baseline_perplexity:.2f}) w4a16')

    ax1.plot(layer_ids, perplexities, marker='o', linestyle='-', color='b')
    ax1.set_ylabel('Perplexity (PPL)')
    ax1.set_title('Perplexity Score When Quantizing an Entire Layer (W4A8KV8)')
    ax1.grid(True)
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # ax2.plot(layer_ids, inference_times, marker='s', linestyle='--', color='r')
    # ax2.set_xlabel('Layer ID')
    # ax2.set_ylabel('Inference Time (ms/token)')
    # ax2.set_title('Inference Time When Quantizing an Entire Layer (W4A8KV8)')
    # ax2.grid(True)
    # layer_ids.remove("W4A16_baseline")  # Remove baseline from layer_ids for x-ticks
    # ax2.set_xticks(np.arange(0, max(layer_ids) + 1, 2))

    plt.tight_layout(pad=2.0)
    save_path = 'sensitivity_analysis_layer_wise_plot_a4_v4.png'
    plt.savefig(save_path, dpi=300)
    print(f"Plot saved as '{save_path}'")
    plt.show()

def plot_module_wise(data):
    """Generates and saves a module-wise sensitivity plot."""
    print("Generating module-wise plot...")
    baseline_perplexity = None
    baseline_inference_time = None
    module_results = defaultdict(lambda: {'layer_ids': [], 'perplexities': [], 'inference_times': []})
    pattern = re.compile(r"model\.layers\.(\d+)\.(self_attn|mlp)\.(\w+)")

    for item in data:
        module_name = item['module_name']
        if module_name == "W4A16_baseline":
            baseline_perplexity = item['perplexity']
            baseline_inference_time = item['inference_time_ms_per_token']
            continue
        
        match = pattern.match(module_name)
        if match:
            layer_id = int(match.group(1))
            module_key = f"{match.group(2)}.{match.group(3)}"
            module_results[module_key]['layer_ids'].append(layer_id)
            module_results[module_key]['perplexities'].append(item['perplexity'])
            module_results[module_key]['inference_times'].append(item['inference_time_ms_per_token'])

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax1 = plt.subplots(1, 1, figsize=(14, 7), sharex=True)
    
    module_types = sorted(module_results.keys())
    colors = plt.cm.tab10(np.linspace(0, 1, len(module_types)))
    markers = ['o', 's', 'X', 'D', '^', 'v', 'P', '*', '<', '>']

    if baseline_perplexity is not None:
        ax1.axhline(y=baseline_perplexity, color='k', linestyle='--', label=f'Baseline PPL ({baseline_perplexity:.2f}) w4a16')

    for i, (module_key, values) in enumerate(module_results.items()):
        ax1.scatter(values['layer_ids'], values['perplexities'], label=module_key, marker=markers[i % len(markers)], color=colors[i % len(colors)], alpha=0.8)
    
    ax1.set_ylabel('Perplexity (PPL)')
    ax1.set_title('Perplexity Score When Quantizing a Single Module (W4A4KV4)', fontsize=16)
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)

    # if baseline_inference_time is not None:
    #     ax2.axhline(y=baseline_inference_time, color='k', linestyle='--', label=f'Baseline Time ({baseline_inference_time:.3f} ms/token)')

    # for i, (module_key, values) in enumerate(module_results.items()):
    #     ax2.scatter(values['layer_ids'], values['inference_times'], label=module_key, marker=markers[i % len(markers)], color=colors[i % len(colors)], alpha=0.8)

    # ax2.set_xlabel('Layer ID', fontsize=12)
    # ax2.set_ylabel('Inference Time (ms/token)')
    # ax2.set_title('Inference Time When Quantizing a Single Module (W4A4KV4)', fontsize=16)
    # ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
    # if any(module_results):
    #     ax2.set_xticks(np.arange(0, max(max(v['layer_ids']) for v in module_results.values() if v['layer_ids']) + 1, 2))

    plt.tight_layout(rect=[0, 0, 0.85, 1])
    save_path = 'sensitivity_analysis_module_wise_a4_v4_plot.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved as '{save_path}'")
    plt.show()

if __name__ == '__main__':
    # --- Set up argument parser ---
    parser = argparse.ArgumentParser(
        description='Plot sensitivity analysis results from a JSON file.'
    )
    
    # Required positional argument for the file path
    parser.add_argument(
        'file_path', 
        type=str, 
        help='The path to the input JSON file.'
    )
    
    # Optional argument for the plot level/granularity
    parser.add_argument(
        '--level', 
        type=str, 
        default='module', 
        choices=['module', 'layer'],
        help="The granularity of the plot: 'module' or 'layer'. Defaults to 'module'."
    )
    
    # Parse the arguments from the command line
    args = parser.parse_args()
    
    # Call the main function with the parsed arguments
    plot_sensitivity_results(args.file_path, args.level)