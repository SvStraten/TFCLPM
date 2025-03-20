import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import tensorflow as tf
import logging

from Model.tfclpm import TFCLPM, preprocess
from Model.sampler import Sampler
from Model.experiments import experiment
from Model.evaluation import save_future_losses_to_csv, save_results_to_csv, save_distribution_to_csv

from edbn.Methods.SDL.sdl import transform_data
import matplotlib.pyplot as plt
import os, sys, time
from sklearn.metrics import accuracy_score, f1_score
import argparse

import time
import pandas as pd
from Data.data import Data
from edbn.Utils.LogFile import LogFile
import edbn.Predictions.setting as setting
from edbn import Methods

def main(method, file, recent_buffer_size, hard_buffer_size, history_buffer_size, history_buffer, MAS_weight):

    if method == 'TFCLPM':
        print("Testing method: TFCLPM")
        dataName, data_sampler, basic_model, _ = preprocess(file)
        learning_object = TFCLPM(verbose=False,
                                seed=123,
                                dev='cpu',
                                dim=4,
                                hidden_units=100,
                                learning_rate=0.005,
                                ntasks=1,
                                gradient_steps=10,
                                loss_window_length=5,
                                loss_window_mean_threshold=0.2,
                                loss_window_variance_threshold=0.1,                                                         
                                MAS_weight=MAS_weight,
                                recent_buffer_size=recent_buffer_size,
                                hard_buffer_size=hard_buffer_size,
                                history_buffer=history_buffer,
                                history_buffer_size=history_buffer_size,
                                model=basic_model)
        
    start_time = time.time()

    tags=[
        'Online Continual',
        # 'Online Continual No Hardbuffer',
        # 'Online No Hardbuffer',
        # 'Online' 
        ]
    
    future_losses, prediction_results, distribution = experiment(data_sampler, learning_object, tags)

    # End measuring the running time
    end_time = time.time()
    
    # Extract actual and predicted labels
    actual_labels = prediction_results['actual_labels']
    predicted_labels = prediction_results['prediction_labels']

    accuracy = round(accuracy_score(actual_labels, predicted_labels), 4)
    macro_f1 = round(f1_score(actual_labels, predicted_labels, average='macro'), 4)

    running_time = end_time - start_time

    # Print accuracy and macro F1-score and running time
    print(f"Model Accuracy: {accuracy * 100}%")
    print(f"Macro F1 Score: {macro_f1}")
    print(f"Running time: {running_time} seconds")

    return accuracy, running_time, dataName, future_losses, distribution, macro_f1, recent_buffer_size, hard_buffer_size, history_buffer_size, history_buffer, MAS_weight

def get_args():
    """Parse command-line arguments with standard values as defaults."""
    parser = argparse.ArgumentParser(description="Run experiments with different configurations.")

    parser.add_argument("--dataset", type=str, default="Data/RecurrentRequest.csv", help="Path to the dataset (CSV file).")
    parser.add_argument("--method", type=str, default="TFCLPM", help="Prediction method to use.")

    parser.add_argument("--recent_buffer_size", type=int, default=500, help="Recent buffer size.")
    parser.add_argument("--hard_buffer_size", type=int, default=100, help="Hard buffer size.")
    parser.add_argument("--history_buffer_size", type=int, default=300, help="History buffer size.")
    parser.add_argument("--MAS_weight", type=float, default=0.5, help="MAS weight.")
    parser.add_argument("--history_buffer", type=bool, default=True, help="Whether to use history buffer (True/False).")
    parser.add_argument("--repetitions", type=int, default=5, help="Number of repetitions.")

    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()

    # List to store results
    results = []
    future_losses_dict = {}

    # Run experiment for specified repetitions
    for repetition in range(1, args.repetitions + 1):
        accuracy, running_time, dataName, future_losses, distribution, macro_f1, recent_buffer_size, hard_buffer_size, history_buffer_size, history_buffer, MAS_weight = main(
            args.method,
            args.dataset, 
            args.recent_buffer_size, 
            args.hard_buffer_size, 
            args.history_buffer_size, 
            args.history_buffer, 
            args.MAS_weight
        )

        # Store results
        results.append([
            args.method, dataName, running_time, accuracy, macro_f1, 
            args.recent_buffer_size, args.hard_buffer_size, 
            args.history_buffer_size, args.history_buffer, args.MAS_weight, repetition
        ])

        # Store future_losses for each method
        future_losses_dict[args.method] = future_losses
        save_future_losses_to_csv(dataName, args.repetitions, future_losses_dict)

    # Save general results to CSV
    save_results_to_csv(results, dataName)
    
    # Save the distribution to csv
    save_distribution_to_csv(distribution, filename=f"Results/results/{dataName}/distribution_results_{history_buffer}.csv")
    
    #Done










 