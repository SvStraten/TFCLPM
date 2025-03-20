import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
import os
from Model.sampler import Sampler
from edbn.Methods.SDL.sdl import train, update, test
from edbn.Utils.LogFile import LogFile
from Data.data import Data
from edbn import Methods
import time

import time
import numpy as np
import tensorflow as tf
from collections import Counter
import copy

class IncrementalUpdateW1():
    def __init__(self,
                 model=None,
                 monthly_batches=None,
                 daily_batches=None,
                 recent_buffer_size=None):
        
        self.model = model
        self.monthly_batches = monthly_batches
        self.daily_batches = daily_batches
        self.recent_buffer_size = recent_buffer_size

    def method(self, data, use_hard_buffer=None, continual_learning=None):
        
        # Tracking variables
        prediction_results = {}
        timings = []
        current_month = 0
        prediction_labels = []
        actual_labels = []
        future_losses = []
        recent_buffer = []
        update_times = 0
        processed_events = 0  

        monthly_batch_keys = sorted(self.monthly_batches.keys())  # Sorted list of months
        print(monthly_batch_keys)
        monthly_event_counts = {month: len(self.monthly_batches[month]['data'].get_data()) for month in monthly_batch_keys}

        # Compute cumulative event indices for monthly updates
        cumulative_month_indices = {}
        running_total = 0

        for month, count in monthly_event_counts.items():
            running_total += count
            cumulative_month_indices[running_total] = month  # Maps cumulative event index to month

        print(f"Monthly batch boundaries: {cumulative_month_indices}")

        basic_model = self.model

        for s in range(len(data.test_inputs[1][self.recent_buffer_size:])):
            recent_buffer.append({'state': data.test_inputs[1][s], 'trgt': data.test_labels[1][s]})
            processed_events += 1  

            if len(recent_buffer) > self.recent_buffer_size:
                del recent_buffer[0]

            if len(recent_buffer) == self.recent_buffer_size:
                x = np.asarray([_['state'] for _ in recent_buffer])
                y = np.asarray([_['trgt'] for _ in recent_buffer])

                xf = x[:]
                yf = y[:]
                yf_pred = basic_model.predict([tf.convert_to_tensor(np.asarray(row).reshape(-1,1)) for row in xf.T.tolist()])
                
                prediction_labels.extend(np.argmax(yf_pred, axis=1).tolist())
                actual_labels.extend(y.tolist())
                accuracy = np.mean(np.argmax(yf_pred, axis=1) == yf)
                print(f"Window accuracy: {accuracy:.4f}")
                future_losses.append(((s+250), accuracy))

                recent_buffer = []
                update_times += 1
                print(f"{update_times}th evaluation done")
                print("------------------")

            if processed_events in cumulative_month_indices:
                month_to_update = cumulative_month_indices[processed_events]
                print(f"Month {month_to_update} completed. Updating model...")

                start_time = time.time()

                m = Methods.get_prediction_method("SDL")

                basic_model = m.update(basic_model, self.monthly_batches[month_to_update]['data'])
                
                timings.append(time.time() - start_time)
                print(f"Model updated after month {month_to_update}")

                current_month += 1

            prediction_results['actual_labels'] = actual_labels
            prediction_results['prediction_labels'] = prediction_labels

        return future_losses, prediction_results, None


class IncrementalUpdateLastDrift():
    def __init__(self,
                 model=None,
                 recent_buffer_size=None,
                 monthly_batches=None,
                 daily_batches=None,
                 dataName=None):
        
        self.model = model
        self.recent_buffer_size = recent_buffer_size
        self.daily_batches = daily_batches
        self.monthly_batches = monthly_batches  
        self.dataName = dataName

    def method(self, data, use_hard_buffer=None, continual_learning=None):
        
        # Tracking variables
        prediction_results = {}
        timings = []
        prediction_labels = []
        actual_labels = []
        future_losses = []
        recent_buffer = []
        update_times = 0
        processed_events = 0  

        basic_model = self.model
        dataName = self.dataName

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
            "Helpdesk": [1000, 2000, 10000]
        }

        last_drift_month = None

        drift_indices = drift_data.get(dataName, [])
        print('Drift Indices: ', drift_indices)

        monthly_batch_keys = sorted(self.monthly_batches.keys()) 
        monthly_event_counts = {month: len(self.monthly_batches[month]['data'].get_data()) for month in monthly_batch_keys}

        cumulative_month_indices = {} 
        running_total = 0

        for month, count in monthly_event_counts.items():
            running_total += count
            cumulative_month_indices[running_total] = month

        print(f"Monthly batch boundaries: {cumulative_month_indices}")

        last_drift_month = monthly_batch_keys[0]  

        for s in range(len(data.test_inputs[1][self.recent_buffer_size:])):
    
            new_event = {'state': data.test_inputs[1][s], 'trgt': data.test_labels[1][s]}
            recent_buffer.append(new_event)

            processed_events += 1  

            if len(recent_buffer) > self.recent_buffer_size:
                del recent_buffer[0]
                
            if len(recent_buffer) == self.recent_buffer_size:
                x = np.asarray([_['state'] for _ in recent_buffer])
                y = np.asarray([_['trgt'] for _ in recent_buffer])

                yf_pred = basic_model.predict([tf.convert_to_tensor(np.asarray(row).reshape(-1,1)) for row in x.T.tolist()])
                
                prediction_labels.extend(np.argmax(yf_pred, axis=1).tolist())
                actual_labels.extend(y.tolist())
                accuracy = np.mean(np.argmax(yf_pred, axis=1) == y)
                print(f"Window accuracy: {accuracy:.4f}")
                future_losses.append(((s+250), accuracy))

                recent_buffer = []
                update_times += 1

            if processed_events in drift_indices:
                print(f"Drift detected at index {processed_events}. Resetting data collection window...")

                for event_idx, month in cumulative_month_indices.items():
                    if processed_events <= event_idx:
                        last_drift_month = month 
                        break

                print(f"Drift detected in month: {last_drift_month}")

            if processed_events in cumulative_month_indices:
                current_month = cumulative_month_indices[processed_events]
                print(f"Month {current_month} completed. Updating model...")

                start_idx = monthly_batch_keys.index(last_drift_month)
                end_idx = monthly_batch_keys.index(current_month)
                months_to_use = monthly_batch_keys[start_idx:end_idx + 1]

                print(f"Using data from months: {months_to_use}")

                start_time = time.time()
                m = Methods.get_prediction_method("SDL")

                combined_log_data = None

                for month in months_to_use:
                    month_data = self.monthly_batches[month]['data']
                    if combined_log_data is None:
                        combined_log_data = month_data
                    else:
                        combined_log_data = combined_log_data.extend_data(month_data)  

                if combined_log_data and combined_log_data.get_data().shape[0] > 0:
                    basic_model = m.update(basic_model, combined_log_data)

                    timings.append(time.time() - start_time)
                    print(f"Model updated at month {current_month} in {timings[-1]:.4f} seconds")


            # Store results
            prediction_results['actual_labels'] = actual_labels
            prediction_results['prediction_labels'] = prediction_labels

        return future_losses, prediction_results, None










