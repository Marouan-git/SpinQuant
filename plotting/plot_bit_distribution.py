import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_bit_distribution(data_folder="./bit_distribution/llama2-7b/report", granularity_filter="module"):
    """
    Analyzes bit distribution from JSON files and generates comparison plots.

    Args:
        data_folder (str): The name of the folder containing the JSON files.
        granularity_filter (str): The granularity to focus on (e.g., 'layer' or 'module').
    """
    all_data = []
    for filename in os.listdir(data_folder):
        if filename.endswith(".json"):
            parts = filename.replace(".json", "").split("_")
            if len(parts) >= 3:
                granularity = parts[-3]
                metric = parts[-2]
                budget = float(parts[-1])

                if granularity == granularity_filter:
                    filepath = os.path.join(data_folder, filename)
                    with open(filepath, 'r') as f:
                        data = json.load(f)
                        for bits, percentage in data.items():
                            all_data.append({
                                "granularity": granularity,
                                "metric": metric,
                                "budget": budget,
                                "bits": int(bits),
                                "percentage": percentage
                            })

    if not all_data:
        print(f"No data found for granularity '{granularity_filter}' in '{data_folder}'.")
        return

    df = pd.DataFrame(all_data)

    # Sort by budget and metric for consistent plotting
    df = df.sort_values(by=["budget", "metric", "bits"])

    budgets = sorted(df["budget"].unique())

    for budget in budgets:
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(12, 7))

        subset_df = df[df["budget"] == budget]

        sns.barplot(data=subset_df, x="metric", y="percentage", hue="bits", ax=ax, palette="viridis")

        ax.set_title(f"Bit Distribution Comparison for Budget: {budget} (Granularity: {granularity_filter})", fontsize=16)
        ax.set_xlabel("Metric", fontsize=12)
        ax.set_ylabel("Percentage (%)", fontsize=12)
        ax.legend(title="Bits")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        # Add percentage labels on top of the bars
        for container in ax.containers:
            ax.bar_label(container, fmt='%.1f%%', label_type='edge', fontsize=9, padding=2)

        # Adjust y-axis to make space for labels
        ax.set_ylim(0, df['percentage'].max() * 1.15)


        output_filename = f"bit_distribution_comparison_{granularity_filter}_budget_{str(budget).replace('.', '_')}.png"
        plt.savefig(output_filename)
        print(f"Plot saved as {output_filename}")
        plt.close(fig)

if __name__ == "__main__":
    # Assuming your JSON files are in a subfolder named 'data'
    # and you are interested in the 'layer' granularity.
    plot_bit_distribution(data_folder="./bit_distribution/llama2-7b/report", granularity_filter="module")