import json
import matplotlib.pyplot as plt
import numpy as np
import re
from collections import defaultdict
import argparse
import os

def plot_sensitivity_results(json_file_path, level='module'):
    """
    Reads top-k sensitivity analysis results from a JSON file and generates plots.
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
        print(f"Error: Invalid plot level '{level}'. Please choose 'module' or 'layer'.")

def plot_layer_wise(data):
    """Generates and saves a separate layer-wise top-k sensitivity plot for each metric."""
    print("Generating layer-wise plots for top-k metrics...")

    # Define the metrics to plot
    metrics_to_plot = {
        'hybrid_score': 'Hybrid Score',
        'normalized_rank_instability': 'Normalized Rank Instability',
        'normalized_confidence_shift': 'Normalized Confidence Shift'
    }

    layer_ids = [item['layer_id'] for item in data]

    # Generate a separate plot for each metric
    for metric_key, metric_name in metrics_to_plot.items():
        print(f"  - Plotting {metric_name}...")
        scores = [item[metric_key] for item in data]

        plt.style.use('seaborn-v0_8-whitegrid')
        fig = plt.figure(figsize=(12, 7))
        ax = fig.add_subplot(111)

        ax.plot(layer_ids, scores, marker='o', linestyle='-', label=metric_name)
        ax.set_xlabel('Layer ID', fontsize=12)
        ax.set_ylabel(metric_name, fontsize=12)
        ax.set_title(f'Top-k Instability: {metric_name} per Layer', fontsize=16)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        
        if layer_ids:
            ax.set_xticks(np.arange(0, max(layer_ids) + 1, 2))

        plt.tight_layout()
        save_path = f'topk_sensitivity_layer_wise_{metric_key}.png'
        plt.savefig(save_path, dpi=300)
        print(f"    Plot saved as '{save_path}'")
        plt.close(fig) # Close the figure to free memory

def plot_module_wise(data):
    """Generates and saves a separate module-wise top-k sensitivity plot for each metric."""
    print("Generating module-wise plots for top-k metrics...")

    # Define the metrics to plot
    metrics_to_plot = {
        'hybrid_score': 'Hybrid Score',
        'normalized_rank_instability': 'Normalized Rank Instability',
        'normalized_confidence_shift': 'Normalized Confidence Shift'
    }
    
    # --- Parse all data first ---
    module_results = defaultdict(lambda: {'layer_ids': [], 'scores': defaultdict(list)})
    pattern = re.compile(r"model\.layers\.(\d+)\.(self_attn|mlp)\.(\w+)")

    for item in data:
        if 'module_name' not in item:
            continue
        match = pattern.match(item['module_name'])
        if match:
            layer_id = int(match.group(1))
            module_key = f"{match.group(2)}.{match.group(3)}"
            module_results[module_key]['layer_ids'].append(layer_id)
            for metric_key in metrics_to_plot.keys():
                if metric_key in item:
                    module_results[module_key]['scores'][metric_key].append(item[metric_key])
    
    if not module_results:
        print("No module data matched the expected format. No plots generated.")
        return

    # --- Generate a separate plot for each metric ---
    for metric_key, metric_name in metrics_to_plot.items():
        print(f"  - Plotting {metric_name}...")
        
        plt.style.use('seaborn-v0_8-whitegrid')
        fig = plt.figure(figsize=(16, 9))
        ax = fig.add_subplot(111)

        module_types = sorted(module_results.keys())
        colors = plt.cm.tab10(np.linspace(0, 1, len(module_types)))
        markers = ['o', 's', 'X', 'D', '^', 'v', 'P', '*', '<', '>']

        for i, module_key in enumerate(module_types):
            values = module_results[module_key]
            # Check if there's score data for the current metric key
            if metric_key in values['scores']:
                ax.scatter(values['layer_ids'], values['scores'][metric_key], label=module_key, 
                            marker=markers[i % len(markers)], color=colors[i % len(colors)], alpha=0.8)
        
        ax.set_xlabel('Layer ID', fontsize=12)
        ax.set_ylabel(metric_name, fontsize=12)
        ax.set_title(f'Top-k Instability: {metric_name} per Module', fontsize=16)
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        
        all_layer_ids = [lid for v in module_results.values() for lid in v['layer_ids']]
        if all_layer_ids:
             ax.set_xticks(np.arange(0, max(all_layer_ids) + 1, 2))

        plt.tight_layout(rect=[0, 0, 0.85, 1])
        save_path = f'topk_sensitivity_module_wise_{metric_key}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"    Plot saved as '{save_path}'")
        plt.close(fig) # Close the figure to free memory

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Plot top-k sensitivity analysis results from a JSON file. Generates a separate plot for each metric.'
    )
    
    parser.add_argument(
        'file_path', 
        type=str, 
        help='The path to the input JSON file.'
    )
    
    parser.add_argument(
        '--level', 
        type=str, 
        default='module', 
        choices=['module', 'layer'],
        help="The granularity of the plot: 'module' or 'layer'. Defaults to 'module'."
    )
    
    args = parser.parse_args()
    
    plot_sensitivity_results(args.file_path, args.level)