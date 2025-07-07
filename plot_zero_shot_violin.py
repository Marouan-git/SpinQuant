import os
import json
import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import glob
from collections import defaultdict
# For path_effects to outline text
from matplotlib import patheffects as pe

def create_violin_plot(data_directory, output_filename="accuracy_variance_violin_plot.png"):
    """
    Generates a violin plot to visualize accuracy variance.
    - Includes an overall average across tasks.
    - Displays the mean value of each violin's data as text, positioned on top of the violin.
    - Always shows a legend for model configurations.

    Args:
        data_directory (str): Path to the directory containing JSON result files.
        output_filename (str): Name of the file to save the plot.
    """
    all_plot_data = []
    json_pattern = os.path.join(data_directory, '*.json')
    json_files = glob.glob(json_pattern)

    if not json_files:
        print(f"No JSON files found in directory: {data_directory}")
        return

    print(f"Found {len(json_files)} JSON files to process: {json_files}")
    
    collected_nb_evals = set()

    for file_path in json_files:
        model_name = os.path.splitext(os.path.basename(file_path))[0]
        model_accuracies_per_task = defaultdict(list)

        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            for task_name, task_data in data.items():
                if isinstance(task_data, dict) and "accuracies" in task_data:
                    accuracies = task_data["accuracies"]
                    if isinstance(accuracies, list) and accuracies:
                        for acc in accuracies:
                            all_plot_data.append({
                                "model_config": model_name,
                                "task": task_name,
                                "accuracy": acc
                            })
                        model_accuracies_per_task[task_name].extend(accuracies)
                        
                        if "nb_evals" in task_data:
                            try:
                                collected_nb_evals.add(int(task_data["nb_evals"]))
                            except ValueError:
                                print(f"Warning: Could not parse nb_evals value '{task_data['nb_evals']}' in {task_name} from {file_path}")
                    else:
                        print(f"Warning: 'accuracies' in task '{task_name}' from file '{file_path}' is not a non-empty list. Skipping for this task.")
                else:
                    print(f"Warning: Task '{task_name}' in file '{file_path}' does not conform to expected structure. Skipping.")
        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from file: {file_path}")
            continue
        except Exception as e:
            print(f"An unexpected error occurred while processing file {file_path}: {e}")
            continue

        if model_accuracies_per_task:
            max_runs_this_model = 0
            if model_accuracies_per_task.values():
                 max_runs_this_model = max(len(acc_list) for acc_list in model_accuracies_per_task.values() if acc_list)

            if max_runs_this_model > 0:
                overall_avg_accuracies_for_model = []
                for i in range(max_runs_this_model):
                    current_run_task_accuracies = []
                    for task_name_key in model_accuracies_per_task:
                        if i < len(model_accuracies_per_task[task_name_key]):
                            current_run_task_accuracies.append(model_accuracies_per_task[task_name_key][i])
                    
                    if current_run_task_accuracies:
                        average_for_run = sum(current_run_task_accuracies) / len(current_run_task_accuracies)
                        overall_avg_accuracies_for_model.append(average_for_run)
                
                for avg_acc_value in overall_avg_accuracies_for_model:
                    all_plot_data.append({
                        "model_config": model_name,
                        "task": "Overall Average",
                        "accuracy": avg_acc_value
                    })

    if not all_plot_data:
        print("No accuracy data found to plot after processing all files.")
        return

    df = pd.DataFrame(all_plot_data)

    title_str = "Accuracy Variance for Zero-Shot Tasks"
    if collected_nb_evals:
        sorted_nb_evals = sorted(list(collected_nb_evals))
        if len(sorted_nb_evals) == 1:
            title_str += f" (nb_evals: {sorted_nb_evals[0]})"
        else:
            title_str += f" (nb_evals: {', '.join(map(str, sorted_nb_evals))})"
    
    task_order = sorted(df['task'].unique().tolist())
    if "Overall Average" in task_order:
        task_order.remove("Overall Average")
        task_order.insert(0, "Overall Average")

    plt.figure(figsize=(max(12, 2.5 * df['task'].nunique()), 8))
    
    ax = sns.violinplot(
        x="task",
        y="accuracy",
        hue="model_config",
        data=df,
        order=task_order,
        cut=0,
        inner="box",
        palette="Blues",
    )

    # Add mean value annotations, positioned on top of each violin
    model_configs_sorted = sorted(df['model_config'].unique().tolist())
    n_models = len(model_configs_sorted)
    
    # Calculate a dynamic y-offset for the text based on the overall data range of the y-axis
    if not df['accuracy'].empty:
        plot_ymin_data, plot_ymax_data = df['accuracy'].min(), df['accuracy'].max()
        y_range_data = plot_ymax_data - plot_ymin_data
        if y_range_data == 0: # Handle case where all accuracies are the same
            y_text_padding = 0.02 # Small absolute padding if no range
        else:
            y_text_padding = y_range_data * 0.025 # 2.5% of the data's y-range for padding
    else:
        y_text_padding = 0.02 # Default padding if DataFrame is empty for some reason

    for i_task, current_task_name in enumerate(task_order):
        for i_model, current_model_name in enumerate(model_configs_sorted):
            subset_df = df[(df['task'] == current_task_name) & (df['model_config'] == current_model_name)]
            if subset_df.empty:
                continue
            
            # if current_task_name == "Overall Average":
            #     print(f"Overall avg mean: {subset_df['accuracy'].mean()}")
            #     print(f"Overall avg std: {subset_df['accuracy'].std()}")
            # elif current_task_name == "hellaswag":
            #     print(f"HellaSwag mean: {subset_df['accuracy'].mean()}")
            #     print(f"HellaSwag std: {subset_df['accuracy'].std()}")

            mean_accuracy_to_display = subset_df['accuracy'].mean()
            max_accuracy_in_violin = subset_df['accuracy'].max() # Get max value for positioning
            min_accuracy_in_violin = subset_df['accuracy'].min() # Get min value for positioning
            range_accuracy_in_violin = max_accuracy_in_violin - min_accuracy_in_violin
            standard_deviation_in_violin = subset_df['accuracy'].std()
            
            x_coord_offset_factor = 0.8
            x_coord = i_task + (i_model - (n_models - 1) / 2.0) * (x_coord_offset_factor / n_models if n_models > 0 else 0)
            
            # Y-coordinate for text: position it above the highest data point in the violin
            y_coord_for_text = max_accuracy_in_violin + y_text_padding
            
            ax.text(x_coord, 
                    y_coord_for_text, # Positioned above the violin's max data point
                    f"mean: {mean_accuracy_to_display:.3f}",
                    color='black',
                    ha='center',
                    va='bottom',
                    fontsize=9,
                    fontweight='bold',
                    path_effects=[pe.withStroke(linewidth=0.75, foreground='white')])

    handles, labels = ax.get_legend_handles_labels()
    if handles: 
        ax.legend(handles=handles, labels=labels, title="Model Config", loc="best")
    
    plt.title(title_str, fontsize=14)
    plt.ylabel("Accuracy", fontsize=12)
    plt.xlabel("Task", fontsize=12)
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Adjust layout. May need to adjust top margin if text is very high.
    plt.tight_layout(rect=[0, 0, 1, 0.96]) 

    # After drawing text, ensure y-limits accommodate it.
    # Matplotlib's auto-scaling usually handles this, but we can be explicit if needed.
    # Get current limits
    ymin, ymax = ax.get_ylim()
    # Tentatively find the highest point any text might reach
    # This is a heuristic; true text height depends on font metrics.
    max_text_y = ymax 
    if not df.empty: # Check if df has data before trying to calculate max text y
        # Find the maximum y_coord_for_text used for any text annotation
        # This requires iterating or storing these values. For simplicity,
        # we rely on Matplotlib's auto-scaling or the existing top margin from tight_layout.
        # If text is consistently cut off, we might need to manually extend ymax.
        # For example: ax.set_ylim(ymin, max(ymax, calculated_max_text_y_position + some_buffer))
        pass


    try:
        plt.savefig(output_filename, dpi=300)
        print(f"Plot saved as {output_filename}")
    except Exception as e:
        print(f"Error saving plot: {e}")
    
    # plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a violin plot for accuracy variance from JSON files, including an overall average, mean annotations, and legend."
    )
    parser.add_argument(
        "data_dir",
        type=str,
        help="Directory containing the JSON files with accuracy data."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="accuracy_variance_violin_plot.png",
        help="Filename for the output plot (default: accuracy_variance_violin_plot.png)"
    )
    args = parser.parse_args()

    create_violin_plot(args.data_dir, args.output)