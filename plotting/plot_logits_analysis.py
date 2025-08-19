import json
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_k_discovery_results(json_file_path):
    """
    Reads the top-k discovery results from a JSON file and generates a
    detailed histogram plot with statistics.
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

    # --- 2. Extract data ---
    stats = data.get('statistics', {})
    hist_data = data.get('histogram_data', {})
    config = data.get('configuration', {})
    
    counts = hist_data.get('counts', [])
    bin_edges = hist_data.get('bin_edges', [])

    if not all([stats, counts, bin_edges, config]):
        print("Error: JSON file is missing required keys ('statistics', 'histogram_data', 'configuration').")
        return

    # --- 3. Create the Plot ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 7))

    # Create a bar chart from the histogram data
    bin_centers = 0.5 * (np.array(bin_edges[:-1]) + np.array(bin_edges[1:]))
    ax.bar(bin_centers, counts, width=np.diff(bin_edges), color='skyblue', edgecolor='black', alpha=0.7)

    # --- 4. Add Titles and Labels ---
    model_name = config.get('model', 'Unknown Model')
    threshold = config.get('probability_threshold', 0.95)
    ax.set_title(f'Distribution of Top-K Tokens to Reach {threshold*100:.0f}% Probability Mass\n({model_name})')
    ax.set_xlabel('Number of Tokens (k)')
    ax.set_ylabel('Frequency (Number of Predictions)')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    # --- 5. Add a Text Box with Statistics ---
    stats_text = (
        f"--- Key Statistics ---\n"
        f"Mean k: {stats.get('mean', 0):.2f}\n"
        f"Median k: {stats.get('median', 0):.2f}\n"
        f"Std. Dev: {stats.get('std_dev', 0):.2f}\n\n"
        f"--- Percentiles ---\n"
        f"75th: {stats.get('percentile_75', 0):.0f}\n"
        f"90th: {stats.get('percentile_90', 0):.0f}\n"
        f"95th: {stats.get('percentile_95', 0):.0f}\n"
        f"99th: {stats.get('percentile_99', 0):.0f}"
    )
    
    # Position the text box in the upper right corner
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right', bbox=props)
            
    # Set a logarithmic scale for the y-axis to better see the distribution tail
    ax.set_yscale('log')
    ax.set_ylabel('Frequency (Log Scale)')


    # --- 6. Save and Show the Plot ---
    output_filename = os.path.splitext(os.path.basename(json_file_path))[0] + '.png'
    save_path = os.path.join(os.path.dirname(json_file_path), output_filename)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Plot saved as '{save_path}'")
    plt.show()


if __name__ == '__main__':
    json_path = 'topk_discovery_results/k_discovery_Llama-2-7b-hf_w4_p0.95.json'
    plot_k_discovery_results(json_path)