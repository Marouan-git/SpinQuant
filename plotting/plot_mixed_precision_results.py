import json
import matplotlib.pyplot as plt
import pandas as pd
import os

# --- Configuration ---
# 1. Set the path to your results folder.
DATA_FOLDER = './mixed_precision_results/llama2-7b/multi_choice/wiki2/module_wise/'

# 2. Specify the granularity to plot ('layer' or 'module').
#    The script will only process files matching this granularity.
TARGET_GRANULARITY = 'module'

# 3. Define plotting styles and labels for each metric.
#    The keys MUST match the 'metric' part of your filenames.
#    Example: for '..._layer_fisherlse_0.35.json', the key is 'fisherlse'.
PLOT_STYLES = {
    'ratio': {'label': 'Max-Median Ratio', 'marker': 's', 'linestyle': '--', 'color': 'red'},
    'fisherlse': {'label': 'Fisher Information', 'marker': 'D', 'linestyle': ':', 'color': 'orange'},
    'fgmp': {'label': 'FGMP', 'marker': 'x', 'linestyle': '-', 'color': 'blue'},
    'fgmpblock32': {'label': 'FGMP Block 32', 'marker': 'p', 'linestyle': '-', 'color': 'brown'},
    'topk10': {'label': 'Top-k10', 'marker': 'v', 'linestyle': '-', 'color': 'purple'},
    'ppl': {'label': 'PPL Degradation', 'marker': 'o', 'linestyle': '-', 'color': 'green'},

}

# PLOT_STYLES = {
#     'fgmp': {'label': 'FGMP', 'marker': '<', 'linestyle': '--', 'color': 'orange'},
#     'fgmpblock32': {'label': 'FGMP Block 32', 'marker': 's', 'linestyle': '--', 'color': 'red'},
#     'fgmpblock64': {'label': 'FGMP Block 64', 'marker': 'x', 'linestyle': '-', 'color': 'blue'},
#     'fgmpblock128': {'label': 'FGMP Block 128', 'marker': 'v', 'linestyle': '-', 'color': 'purple'},
# }

# PLOT_STYLES = {
#     'topk5': {'label': 'Top-k5', 'marker': '<', 'linestyle': '--', 'color': 'orange'},
#     'topk10': {'label': 'Top-k10', 'marker': 's', 'linestyle': '--', 'color': 'red'},
#     'topk20': {'label': 'Top-k20', 'marker': 'x', 'linestyle': '-', 'color': 'blue'},
#     'topk50': {'label': 'Top-k50', 'marker': 'v', 'linestyle': '-', 'color': 'purple'},
# }


# 4. Set the baseline perplexity for the reference line.
BASE_PERPLEXITY = 5.651648044586182 # Wiki2
#BASE_PERPLEXITY = 6.58 # Llama-3-8B on Wiki2
#BASE_PERPLEXITY = 8.4345 # Llama-3.2-3B on Wiki2
#BASE_PERPLEXITY = 7.50929594039917 # c4q

# --- Data Processing ---
results = []
print(f"🔍 Searching for JSON files in '{DATA_FOLDER}'...")

if not os.path.isdir(DATA_FOLDER):
    print(f"❌ Error: Directory not found at '{DATA_FOLDER}'. Please check the DATA_FOLDER path.")
else:
    # List all files in the specified directory
    files = [f for f in os.listdir(DATA_FOLDER) if f.endswith('.json')]

    for file_name in files:
        try:
            # Robustly parse filename from the end
            parts = file_name.replace('.json', '').split('_')
            granularity = parts[-3]
            metric = parts[-2]
            budget = float(parts[-1])

            # Process only files that match the target granularity
            if granularity == TARGET_GRANULARITY:
                # Read the perplexity from the JSON file
                with open(os.path.join(DATA_FOLDER, file_name), 'r') as f:
                    data = json.load(f)
                    perplexity = data['perplexity']

                results.append({
                    'budget': budget,
                    'metric': metric,
                    'perplexity': perplexity
                })

        except (IndexError, ValueError) as e:
            print(f"⚠️ Could not process file '{file_name}': {e}. Skipping.")

# --- Plot Generation ---
if not results:
    print(f"\n❌ No data found for granularity '{TARGET_GRANULARITY}'. No plot will be generated.")
else:
    # Convert results to a pandas DataFrame for easy manipulation
    df = pd.DataFrame(results)

    # Create the plot
    plt.figure(figsize=(12, 7))

    # Get the list of unique metrics found in the data
    metrics_in_data = df['metric'].unique()
    print(f"\n📊 Found data for metrics: {', '.join(metrics_in_data)}")

    for metric in metrics_in_data:
        if metric in PLOT_STYLES:
            # Filter data for the current metric and sort by budget for correct line plotting
            metric_data = df[df['metric'] == metric].sort_values(by='budget')
            style = PLOT_STYLES[metric]

            plt.plot(
                metric_data['budget'],
                metric_data['perplexity'],
                marker=style['marker'],
                linestyle=style['linestyle'],
                label=style['label'],
                color=style['color']
            )
        else:
            print(f"🤔 Warning: Metric '{metric}' found in files but has no defined style in PLOT_STYLES. It will be skipped.")

    # Add titles and labels for clarity
    plt.title(f'Perplexity vs. Budget', fontsize=16)
    plt.xlabel('BOPs Budget Multiplicative', fontsize=12)
    plt.ylabel('Perplexity (Lower is Better)', fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Add the black dashed line for base perplexity
    plt.axhline(y=BASE_PERPLEXITY, color='black', linestyle='--', label=f'Base W4A16KV16 Perplexity')
    plt.text(
        x=df['budget'].min(), y=BASE_PERPLEXITY * 0.9995, s=f'{BASE_PERPLEXITY:.4f}',
        color='black', ha='left', va='bottom', fontsize=10
    )

    # Add legend
    plt.legend(fontsize=12)
    
    # Save the plot to a file
    output_filename = f'perplexity_vs_budget_{TARGET_GRANULARITY}.png'
    plt.savefig(output_filename)

    print(f"\n✅ Plot successfully generated and saved as '{output_filename}'")