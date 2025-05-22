import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import random # For dummy data generation

# Define the list of files and their desired labels for the x-axis
file_configurations = [
    {"path": "eval_results/llama2-7b-4_4_4-no_had.json", "label": "no_had"},
    {"path": "eval_results/llama2-7b-4_4_4-sparse-had-p10.json", "label": "sparse-p10"},
    {"path": "eval_results/llama2-7b-4_4_4-sparse-had-p25.json", "label": "sparse-p25"},
    {"path": "eval_results/llama2-7b-4_4_4-sparse-had-p50.json", "label": "sparse-p50"},
    {"path": "eval_results/llama2-7b-4_4_4-sparse-had-p75.json", "label": "sparse-p75"},
    {"path": "eval_results/llama2-7b-4_4_4-had.json", "label": "had"},
]



# Prepare data for plots
violin_plot_data_list = []
config_to_avg_ppl = {}
ordered_config_labels = [] # To maintain the order of configurations as defined

for config in file_configurations:
    file_path = config["path"]
    label = config["label"]
    # Ensure label is added to ordered_config_labels only once and in order
    if label not in ordered_config_labels:
        ordered_config_labels.append(label)

    try:
        with open(file_path, 'r') as f:
            data = json.load(f)

        # Data for violin plot (list_token_time)
        cuda_timings = data.get("cuda_timing_ms", {})
        list_token_time = cuda_timings.get("list_token_time")

        if list_token_time and isinstance(list_token_time, list):
            for time_val in list_token_time:
                violin_plot_data_list.append({'Configuration': label, 'Token Time (ms)': time_val})
        else:
            print(f"Warning: 'list_token_time' not found or not a list in {file_path} for violin plot.")

        # Data for line plot (avg_ppl)
        avg_ppl_val = data.get("avg_ppl")
        if avg_ppl_val is not None:
            config_to_avg_ppl[label] = avg_ppl_val
        else:
            print(f"Warning: 'avg_ppl' not found in {file_path}.")
            config_to_avg_ppl[label] = None # Store None to maintain order, will be filtered later if needed

    except FileNotFoundError:
        print(f"Error: File not found - {file_path}. Skipping.")
        config_to_avg_ppl[label] = None # Mark as None if file not found
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {file_path}. Skipping.")
        config_to_avg_ppl[label] = None
    except Exception as e:
        print(f"An unexpected error occurred with file {file_path}: {e}. Skipping.")
        config_to_avg_ppl[label] = None


if not violin_plot_data_list:
    print("No data available for violin plot. Exiting.")
else:
    df_violin_plot = pd.DataFrame(violin_plot_data_list)

    # Prepare avg_ppl data in the correct order of configurations
    avg_ppls_for_plot = [config_to_avg_ppl.get(conf_label) for conf_label in ordered_config_labels]
    
    
    # Create the combined plot
    fig, ax1 = plt.subplots(figsize=(14, 8)) # Adjust figure size as needed

    # 1. Violin plot for Token Time on primary y-axis (ax1)
    # Use the 'order' parameter in violinplot to ensure consistent x-axis ordering
    sns.violinplot(x='Configuration', y='Token Time (ms)', data=df_violin_plot,
                   order=ordered_config_labels, palette="Blues", inner="box", ax=ax1)
    ax1.set_xlabel('Configuration', fontsize=14)
    ax1.set_ylabel('Inference time per token (ms) (Distribution)', fontsize=14, color='C0') # C0 is default blue
    ax1.tick_params(axis='y', labelcolor='C0', labelsize=12)
    ax1.tick_params(axis='x', labelsize=12, rotation=30)
    ax1.grid(axis='y', linestyle='--', alpha=0.7, which='major')

    # 2. Line plot for Average PPL on secondary y-axis (ax2)
    ax2 = ax1.twinx() # Create a second y-axis sharing the same x-axis
    
    # Plotting PPL
    # `ordered_config_labels` provides the x-coordinates (categories)
    # `avg_ppls_for_plot` provides the y-coordinates
    line_ppl = ax2.plot(ordered_config_labels, avg_ppls_for_plot, color='red', linestyle='-', marker='o', linewidth=2, markersize=8, label='PPL') # C1 is default orange
    ax2.set_ylabel('PPL', fontsize=14, color='red')
    ax2.tick_params(axis='y', labelcolor='red', labelsize=12)
    


    plt.title('Inference time per token distribution and PPL per configuration - 20 eval runs', fontsize=18, pad=20)

    
    ax2.legend(loc='upper center', bbox_to_anchor=(0.5, 0.98))


    fig.tight_layout() # Adjust layout to prevent overlap

    # Save the plot to a file
    output_filename = "token_time_ppl_combined_plot.png"
    try:
        plt.savefig(output_filename, dpi=300) # dpi for higher resolution
        print(f"Plot saved to {output_filename}")
    except Exception as e:
        print(f"Error saving plot: {e}")

    # Show the plot
    plt.show()

    # Clean up the figure from memory
    plt.close(fig)