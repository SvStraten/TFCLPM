import os
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os
import os
import matplotlib.pyplot as plt
import math
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

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