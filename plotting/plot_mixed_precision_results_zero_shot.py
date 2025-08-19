import json
import matplotlib.pyplot as plt
import pandas as pd
import os

# --- Configuration ---
# 1. Set the path to your zero-shot results folder.
DATA_FOLDER = './mixed_precision_results/llama2-7b/lm_eval/module_wise/'

BASELINE_FILE_NAME = "lm_eval_results_mixed_precision_base_accuracy.json"

# 2. Specify the granularity to plot ('layer' or 'module').
TARGET_GRANULARITY = 'module'

# 3. Define plotting styles and labels for each metric.
#    The keys MUST match the 'metric' part of your filenames.
PLOT_STYLES = {
    'ratio': {'label': 'Max-Median Ratio', 'marker': 's', 'linestyle': '--', 'color': 'red'},
    'fisherlse': {'label': 'Fisher Information (LSE)', 'marker': 'D', 'linestyle': ':', 'color': 'orange'},
    'fgmp': {'label': 'FGMP', 'marker': 'x', 'linestyle': '-', 'color': 'blue'},
    'fgmpblock32': {'label': 'FGMP Block 32', 'marker': 'p', 'linestyle': '-', 'color': 'brown'},
    'topk10': {'label': 'Top-k10', 'marker': 'v', 'linestyle': '-', 'color': 'purple'},
    'ppl': {'label': 'PPL Degradation', 'marker': 'o', 'linestyle': '-', 'color': 'green'},

}

PLOT_STYLES = {
    'fgmp': {'label': 'FGMP', 'marker': '<', 'linestyle': '--', 'color': 'orange'},
    'fgmpblock32': {'label': 'FGMP Block 32', 'marker': 's', 'linestyle': '--', 'color': 'red'},
    'fgmpblock64': {'label': 'FGMP Block 64', 'marker': 'x', 'linestyle': '-', 'color': 'blue'},
    'fgmpblock128': {'label': 'FGMP Block 128', 'marker': 'v', 'linestyle': '-', 'color': 'purple'},
}

PLOT_STYLES = {
    'topk5': {'label': 'Top-k5', 'marker': '<', 'linestyle': '--', 'color': 'orange'},
    'topk10': {'label': 'Top-k10', 'marker': 's', 'linestyle': '--', 'color': 'red'},
    'topk20': {'label': 'Top-k20', 'marker': 'x', 'linestyle': '-', 'color': 'blue'},
    'topk50': {'label': 'Top-k50', 'marker': 'v', 'linestyle': '-', 'color': 'purple'},
}



# --- Data Processing ---
results = []
baseline_accuracies = {} # Dictionary to store base accuracies for each task
print(f"🔍 Searching for JSON files in '{DATA_FOLDER}'...")

if not os.path.isdir(DATA_FOLDER):
    print(f"❌ Error: Directory not found at '{DATA_FOLDER}'. Please check the path.")
else:
    files = [f for f in os.listdir(DATA_FOLDER) if f.endswith('.json')]

    for file_name in files:
         # --- Process the baseline accuracy file ---
        file_path = os.path.join(DATA_FOLDER, file_name)
        if file_name == BASELINE_FILE_NAME:
            print(f"🎯 Found and processing baseline accuracy file: {file_name}")
            with open(file_path, 'r') as f:
                data = json.load(f)
            for task_name, task_data in data.items():
                if isinstance(task_data, dict) and 'mean_accuracy' in task_data:
                    baseline_accuracies[task_name] = task_data['mean_accuracy']
                elif task_name == 'mean_accuracy' and isinstance(task_data, float):
                    baseline_accuracies['overall_mean_accuracy'] = task_data
            continue # Move to the next file
        try:
            # Robustly parse filename from the end
            parts = file_name.replace('.json', '').split('_')
            granularity = parts[-3]
            metric = parts[-2]
            budget = float(parts[-1])

           

            # Process only files that match the target granularity
            if granularity != TARGET_GRANULARITY and granularity != 'block':
                continue

            # Read the JSON content
            with open(os.path.join(DATA_FOLDER, file_name), 'r') as f:
                data = json.load(f)

            # Extract accuracy for each task and the overall mean
            for task_name, task_data in data.items():
                accuracy = None
                # Handle individual tasks which are dictionaries
                if isinstance(task_data, dict) and 'mean_accuracy' in task_data:
                    accuracy = task_data['mean_accuracy']
                    
                # Handle the overall mean_accuracy which is a direct float value
                elif task_name == 'mean_accuracy' and isinstance(task_data, float):
                    accuracy = task_data
                    # Use a consistent name for the overall average task
                    task_name = 'overall_mean_accuracy'

                if accuracy is not None:
                    results.append({
                        'budget': budget,
                        'metric': metric,
                        'task': task_name,
                        'accuracy': accuracy
                    })

        except (IndexError, ValueError, json.JSONDecodeError) as e:
            print(f"⚠️ Could not process file '{file_name}': {e}. Skipping.")

# --- Plot Generation ---
if not results:
    print(f"\n❌ No data found for granularity '{TARGET_GRANULARITY}'. No plots will be generated.")
else:
    df = pd.DataFrame(results)
    all_tasks = df['task'].unique()
    print(f"\n📊 Found data for tasks: {', '.join(all_tasks)}")
    
    # Generate one plot for each task
    for task_name in all_tasks:
        plt.figure(figsize=(12, 7))
        
        task_df = df[df['task'] == task_name]
        metrics_in_task = sorted(task_df['metric'].unique())

        for metric in metrics_in_task:
            if metric in PLOT_STYLES:
                metric_data = task_df[task_df['metric'] == metric].sort_values(by='budget')
                style = PLOT_STYLES[metric]
                plt.plot(
                    metric_data['budget'],
                    metric_data['accuracy'],
                    marker=style.get('marker', 'o'),
                    linestyle=style.get('linestyle', '-'),
                    label=style.get('label', metric),
                    color=style.get('color', 'blue')
                )
            else:
                print(f"🤔 Warning: Metric '{metric}' has no defined style. Skipping on plot for '{task_name}'.")
        
        # --- Add the baseline accuracy line ---
        if task_name in baseline_accuracies:
            base_acc = baseline_accuracies[task_name]
            plt.axhline(
                y=base_acc, 
                color='black', 
                linestyle='--', 
                linewidth=2, 
                label=f'Base Accuracy W4A16KV16 ({base_acc:.4f})'
            )
            plt.text(
                x=task_df['budget'].min(), 
                y=base_acc * 1.0003, 
                s=f'{base_acc:.4f}', 
                color='black', 
                fontsize=10, 
                ha='left'
            )

        # Customize plot appearance
        pretty_task_name = task_name.replace('_', ' ').replace('mean accuracy', 'Mean Accuracy').title()
        plt.title(f'Zero-Shot Accuracy on {pretty_task_name}', fontsize=16)
        plt.xlabel('BOPs Budget Multiplicative', fontsize=12)
        plt.ylabel('Accuracy (Higher is Better)', fontsize=12)
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.legend(fontsize=12)
        
        # Save the plot
        output_filename = f'accuracy_{task_name}_{TARGET_GRANULARITY}.png'
        plt.savefig(output_filename)
        print(f"✅ Plot saved as '{output_filename}'")
        
        # Close the figure to free up memory before creating the next one
        plt.close()

    print("\n🎉 All plots generated successfully.")