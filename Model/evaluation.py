import os
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os
import os
import matplotlib.pyplot as plt
import math
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def plot_accuracy_at_given_index(dataName, repetition, future_losses_dict):
    """
    Plots accuracy over event index for multiple methods and adds vertical red lines at drift indices.

    :param dataName: Name of the dataset.
    :param repetition: Repetition index.
    :param future_losses_dict: Dictionary containing method names as keys and their (event index, accuracy) tuples as values.
    """
    # Hardcoded drift indices for each dataset
    drift_data = {
        "IOR5000": [5212, 10460, 15662, 20990, 26138, 31420, 36583, 41841, 47027, 52311],
        "IRO5000": [5212, 10741, 15949, 21449, 26593, 32154, 37316, 42941, 48123, 53639],
        "OIR5000": [5212, 11033, 16235, 22165, 27314, 33174, 38341, 44256, 49443, 55322],
        "ORI5000": [5212, 10988, 16189, 21939, 27089, 32943, 38111, 43746, 48934, 54604],
        "RIO5000": [5212, 10663, 15868, 21403, 26553, 32105, 37273, 42765, 47953, 53418],
        "ROI5000": [5212, 10354, 15555, 20740, 25888, 31192, 36354, 41555, 46737, 51987],
        "DomesticDeclarations": [9878, 14501],
        "InternationalDeclarations": [12467, 17361],
        "RequestForPayment": [4913, 8298],
        "HelpdeskDrift": [9075, 20655]
    }
    
    # Create plot
    plt.figure(figsize=(10, 5))
    
    # Plot accuracy for each method
    colors = ['b', 'g']  # Different colors for methods
    for idx, (method, future_losses) in enumerate(future_losses_dict.items()):
        event_indices = [entry[0] for entry in future_losses]
        accuracies = [entry[1] for entry in future_losses]
        plt.plot(event_indices, accuracies, linestyle='-', color=colors[idx], label=method)
    
    # Add drift indices as vertical red lines
    if dataName in drift_data:
        drift_indices = drift_data[dataName]
        for drift_idx in drift_indices:
            plt.axvline(x=drift_idx, color='red', linestyle='--', label='Drift' if drift_idx == drift_indices[0] else "_nolegend_")
    
    # Labels and title
    plt.xlabel('Event Index')
    plt.ylabel('Accuracy')
    plt.title(f'Accuracy Over Time for {dataName}')
    plt.legend()
    plt.grid(True)
    
    # Ensure the results directory exists
    os.makedirs(f"Results/results/{dataName}", exist_ok=True)
    
    # Save the figure
    plt.savefig(f'Results/results/{dataName}/{dataName}_accuracy_lastdrift.png', dpi=300)
    plt.close()

def plot_distributions(dataName, repetition, distribution, history_buffer):
    """
    Plots normalized main window distribution and updated hard buffer distribution as bar charts for all update indices 
    in a grid layout with 5 columns.

    :param dataName: Name of the dataset.
    :param repetition: Repetition index.
    :param distribution: List of dictionaries containing class distributions at each update_idx.
    """
    os.makedirs(f"Results/results/{dataName}", exist_ok=True)  # Ensure output directory exists

    num_updates = len(distribution)
    cols = 5  # Set 5 columns
    rows = math.ceil(num_updates / cols)  # Determine number of rows

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows), sharex=True)

    # Flatten axes for easy iteration (handles cases where there are fewer updates than grid cells)
    axes = axes.flatten() if num_updates > 1 else [axes]

    for update_idx, dist in enumerate(distribution):
        ax = axes[update_idx]  # Get the corresponding subplot

        main_window_dist = dist["main_window_distribution"]  # Rolling buffer (500 samples)
        updated_hard_buffer_dist = dist["updated_hard_buffer_distribution"]  # Hard buffer (100 samples)

        # Normalize distributions
        total_main = sum(main_window_dist.values())
        total_updated = sum(updated_hard_buffer_dist.values())

        main_window_norm = {k: v / total_main for k, v in main_window_dist.items()}
        updated_hard_buffer_norm = {k: v / total_updated for k, v in updated_hard_buffer_dist.items()}

        # Get sorted class labels
        all_labels = sorted(set(main_window_norm.keys()).union(updated_hard_buffer_norm.keys()))

        # Extract normalized counts (set to 0 if missing)
        main_counts = [main_window_norm.get(label, 0) for label in all_labels]
        updated_counts = [updated_hard_buffer_norm.get(label, 0) for label in all_labels]

        # Create bar plot
        bar_width = 0.4  # Width of bars
        x_positions = range(len(all_labels))

        ax.bar(x_positions, main_counts, width=bar_width, label="Main Window", alpha=0.7)
        ax.bar([x + bar_width for x in x_positions], updated_counts, width=bar_width, label="Updated Hard Buffer", alpha=0.7)

        # Labels and title
        ax.set_ylabel('Normalized Frequency')
        ax.set_xticks([x + bar_width / 2 for x in x_positions])
        ax.set_xticklabels(all_labels, rotation=45)  # Rotate x-labels for readability
        ax.set_title(f'Update {update_idx}')
        ax.legend()
        ax.grid(axis='y')

    # Hide empty subplots (if any)
    for i in range(num_updates, len(axes)):
        fig.delaxes(axes[i])

    # Set shared x-axis label
    fig.text(0.5, 0.04, 'Class Label', ha='center', fontsize=12)

    # Save the figure
    plt.tight_layout(rect=[0, 0.04, 1, 1])  # Adjust layout to fit x-axis label
    plt.savefig(f'Results/results/{dataName}/{dataName}_distribution_all_updates_{repetition}_{history_buffer}.png', dpi=300)
    plt.close()  # Free memory

def save_future_losses_to_csv(dataName, repetition, future_losses_dict):
    """
    Saves accuracy over event index for multiple methods into a CSV file.

    :param dataName: Name of the dataset.
    :param repetition: Repetition index.
    :param future_losses_dict: Dictionary containing method names as keys and their (event index, accuracy) tuples as values.
    """
    # Prepare a DataFrame to store results
    data_records = []

    for method, future_losses in future_losses_dict.items():
        for event_index, accuracy in future_losses:
            data_records.append([method, event_index, accuracy])

    # Convert to DataFrame
    df = pd.DataFrame(data_records, columns=["method", "index", "accuracy"])

    # Ensure the results directory exists
    results_dir = f"Results/results/{dataName}"
    os.makedirs(results_dir, exist_ok=True)

    # Save to CSV
    csv_filename = f"{results_dir}/{dataName}_accuracy_{repetition}.csv"
    df.to_csv(csv_filename, index=False)

    return df

def save_results_to_csv(results, dataName):
    """
    Saves general results for all methods into a CSV file.
    
    :param results: List containing result data for each experiment.
    :param dataName: Name of the dataset.
    """
    df = pd.DataFrame(results, columns=[
        "method", "dataName", "running_time", "accuracy", "macro_f1", 
        "recent_buffer_size", "hard_buffer_size", "history_buffer_size", 
        "history_buffer", "MAS_weight", "repetition"
    ])
    
    # Ensure the results directory exists
    results_dir = f"Results/results/{dataName}"
    os.makedirs(results_dir, exist_ok=True)
    
    # Save to CSV
    csv_filename = f"{results_dir}/{dataName}_results.csv"
    df.to_csv(csv_filename, index=False)
    
    return df

def save_distribution_to_csv(distribution_data, filename="distribution_results.csv"):
    """
    Saves the distribution data to a CSV file.

    Parameters:
    distribution_data (list of dict): Each dictionary contains the distributions for an update index.
    filename (str): Name of the output CSV file.
    """
    records = []
    
    for idx, dist in enumerate(distribution_data):
        main_window_distribution = dist["main_window_distribution"]
        updated_hard_buffer_distribution = dist["updated_hard_buffer_distribution"]

        labels = list(set(main_window_distribution.keys()).union(set(updated_hard_buffer_distribution.keys())))
        labels.sort()

        for label in labels:
            records.append({
                "Update_Index": idx + 1,
                "Class_Label": label,
                "Window_Distribution": main_window_distribution.get(label, 0),
                "Hard_Buffer_Distribution": updated_hard_buffer_distribution.get(label, 0)
            })

    df = pd.DataFrame(records)
    df.to_csv(filename, index=False)
    print(f"Distribution data saved to {filename}")