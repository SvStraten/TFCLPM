import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import tensorflow as tf
import numpy as np
from collections import Counter
from tensorflow.keras.optimizers import Nadam #type: ignore
import os
import tensorflow as tf
from Model.sampler import Sampler
import pandas as pd
from Data.data import Data
from edbn.Utils.LogFile import LogFile
import edbn.Predictions.setting as setting
from edbn import Methods

def preprocess(file):
    """Preprocesses the event log data from the given CSV file."""

    # Import the data
    data = pd.read_csv(file, low_memory=False)
    numEvents = data.shape[0]
    print("Total number of events:", numEvents)

    # Extract the filename from the file path
    filename = os.path.basename(file)
    dataName = os.path.splitext(filename)[0]
    print("Dataset:", dataName)

    # Create a LogFile object
    d = Data(dataName,
             LogFile(filename=file, delim=",", header=0, rows=None, time_attr="completeTime", trace_attr="case",
                     activity_attr='event', convert=False))
   
    # Keep only relevant attributes
    d.logfile.keep_attributes(['event', 'completeTime'])

    # Select the prediction method
    m = Methods.get_prediction_method("SDL")
    print("Prediction method:", m)
    
    # Prepare the data
    s = setting.STANDARD
    d.prepare(s)
    print("### Data Preprocessed")

    # Train the basic model
    basic_model = m.train(d.get_test_batchi(0, 500))
    print("### Basic model has been trained")

    # Detect the time format
    connect_symbol = "-" if "-" in d.logfile.get_data()['completeTime'][0] else "/"
    
    # Define possible datetime formats
    formats = [
        f"%Y{connect_symbol}%m{connect_symbol}%d %H:%M:%S%z",  # With timezone offset
        f"%Y{connect_symbol}%m{connect_symbol}%d %H:%M:%S",  # Standard format
        f"%Y{connect_symbol}%m{connect_symbol}%d %H:%M:%S.%f"  # With microseconds
    ]

    # Try to identify the correct format
    for timeformat in formats:
        try:
            d.logfile.get_data()['completeTime'] = pd.to_datetime(
                d.logfile.get_data()['completeTime'], format=timeformat, exact=True
            )
            print("The detected time format is:", timeformat)
            break
        except ValueError:
            continue

    # Create batches
    monthly_batches = d.create_batch(split='months', timeformat=timeformat) 

    # Create a sampler
    data_sampler = Sampler(data=d)
    print("### Sampler done")

    return dataName, data_sampler, basic_model, monthly_batches


class TFCLPM:
    def __init__(self,
                 verbose=False,
                 seed=123,
                 dev='cpu',
                 dim=4,
                 hidden_units=100,
                 learning_rate=0.005,
                 ntasks=1,
                 gradient_steps=None,
                 loss_window_length=None,
                 loss_window_mean_threshold=None,
                 loss_window_variance_threshold=None,
                 MAS_weight=None,
                 recent_buffer_size=None,
                 hard_buffer_size=None,
                 history_buffer=None,
                 history_buffer_size=None,
                 model=None):

        # Define the model
        self.model = model
        self.optimizer = Nadam(learning_rate=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, clipvalue=3)
        self.loss_fn = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)

        self.verbose = verbose
        self.dim = dim
        self.ntasks = ntasks
        self.gradient_steps = gradient_steps
        self.loss_window_length = loss_window_length
        self.loss_window_mean_threshold = loss_window_mean_threshold
        self.loss_window_variance_threshold = loss_window_variance_threshold
        self.MAS_weight = MAS_weight
        self.recent_buffer_size = recent_buffer_size
        self.hard_buffer_size = hard_buffer_size
        self.history_buffer = history_buffer
        self.history_buffer_size = history_buffer_size

    def method(self, data, use_hard_buffer=False, continual_learning=False):
        losses = []
        test_loss = {i: [] for i in range(self.ntasks)}
        future_losses = []
        recent_buffer = []
        hard_buffer = []
        loss_window = []
        loss_window_means = []
        loss_window_variances = []
        update_tags = []
        new_peak_detected = True
        star_variables = []
        omegas = []
        update_times = 0
        prediction_results = {}
        prediction_labels = []
        actual_labels = []
        distribution = []
        
        # Initialize history buffer if history_buffer=True
        history_hard_buffer = [] if self.history_buffer else None  

        for t in range(self.ntasks):
            for s in range(len(data.test_inputs[t][500:])):
                # Initialize an empty list to store the prediction results and actual labels
                recent_buffer.append({'state': data.test_inputs[t][s], 'trgt': data.test_labels[t][s]})
                if len(recent_buffer) > self.recent_buffer_size:
                    del recent_buffer[0]

                if len(recent_buffer) == self.recent_buffer_size:
                    msg = 'task: {0} step: {1}'.format(t, s)

                    x = np.asarray([_['state'] for _ in recent_buffer])
                    y = np.asarray([_['trgt'] for _ in recent_buffer])

                    main_window_distribution = Counter(y)
                    print(main_window_distribution)
                    
                    xf=x[:]
                    yf=y[:]
                    yf_pred = self.model.predict([tf.convert_to_tensor(np.asarray(row).reshape(-1,1)) for row in xf.T.tolist()])
                    
                    # Inside the loop where you calculate the accuracy
                    prediction_labels.extend(np.argmax(yf_pred, axis=1).tolist())
                    actual_labels.extend(yf.tolist())
                    accuracy = np.mean(np.argmax(yf_pred, axis=1) == yf)
                    future_losses.append(((s+250), accuracy))

                    if use_hard_buffer and len(hard_buffer) != 0:
                        print('hard buffer is not empty',s)
                        xh = np.asarray([entry[1] for entry in hard_buffer])
                        yh = np.asarray([entry[2] for entry in hard_buffer])
                    
                    for gs in range(self.gradient_steps):
                        with tf.GradientTape() as tape:

                            y_pred = self.model([tf.convert_to_tensor(np.asarray(row).reshape(-1,1)) for row in x.T.tolist()],training=True)
                            y_sup = tf.one_hot(tf.convert_to_tensor(y, dtype=tf.int32), depth=y_pred.shape[1], dtype=tf.float32)
                            recent_loss = self.loss_fn(y_sup, y_pred)
                            total_loss = tf.reduce_sum(recent_loss)

                            if use_hard_buffer and len(hard_buffer) != 0:
                                yh_pred = self.model([tf.convert_to_tensor(np.asarray(row).reshape(-1,1)) for row in xh.T],training=True)
                                yh_sup = tf.one_hot(tf.convert_to_tensor(yh, dtype=tf.int32), depth=y_pred.shape[1], dtype=tf.float32)
                                hard_loss = self.loss_fn(yh_sup, yh_pred)
                                total_loss += tf.reduce_sum(hard_loss)

                            if gs == 0:
                                first_train_loss = total_loss.numpy()

                            if continual_learning and len(star_variables) != 0 and len(omegas) != 0:
                                print('Add MAS regularization')
                                for pindex, p in enumerate(self.model.trainable_variables):
                                    total_loss += self.MAS_weight / 2.0 * tf.reduce_sum(
                                        tf.convert_to_tensor(omegas[pindex], dtype=tf.float32) *
                                        (p - star_variables[pindex]) ** 2)

                            gradients = tape.gradient(total_loss, self.model.trainable_variables)
                            self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

                    
                    if use_hard_buffer and len(hard_buffer) != 0:
                        xt = np.concatenate((x, xh))
                        yt = np.concatenate((y, yh))
                    else:
                        xt = x[:]
                        yt = y[:]

                    yt_pred = self.model.predict([tf.convert_to_tensor(np.asarray(row).reshape(-1,1)) for row in xt.T.tolist()])
                    accuracy = np.mean(np.argmax(yt_pred, axis=1) == yt)
                    msg += ' recent loss: {0:0.3f}'.format(np.mean(recent_loss.numpy()))

                    if use_hard_buffer and len(hard_buffer) != 0:
                        msg += ' hard loss: {0:0.3f}'.format(np.mean(hard_loss.numpy()))
                    losses.append(np.mean(accuracy))

                    # Add loss to loss_window and detect loss plateaus
                    loss_window.append(np.mean(first_train_loss))

                    if len(loss_window) > self.loss_window_length:
                        del loss_window[0]
                    loss_window_mean = np.mean(loss_window)
                    loss_window_variance = np.var(loss_window)

                    if not new_peak_detected and loss_window_mean > last_loss_window_mean + np.sqrt(
                            last_loss_window_variance):
                        new_peak_detected = True

                    if continual_learning and loss_window_mean < self.loss_window_mean_threshold and \
                            loss_window_variance < self.loss_window_variance_threshold and new_peak_detected:
                        print('Start updating importance weights')
                        count_updates += 1
                        update_tags.append(0.01)
                        last_loss_window_mean = loss_window_mean
                        last_loss_window_variance = loss_window_variance
                        

                        gradients = [np.zeros_like(p.numpy()) for p in self.model.trainable_variables]

                        for sx in [_['state'] for _ in hard_buffer]:
                            with tf.GradientTape() as tape:
                                y_pred = self.model([tf.convert_to_tensor([x]) for x in sx])
                                loss = tf.norm(y_pred, ord=2, axis=1)
                            grads = tape.gradient(loss, self.model.trainable_variables)
                            for pindex, p in enumerate(grads):
                                if isinstance(p, tf.IndexedSlices):
                                    p = tf.convert_to_tensor(p)
                                gradients[pindex] += np.abs(p.numpy())

                        omegas_old = omegas[:]
                        omegas = []
                        star_variables = []

                        for pindex, p in enumerate(self.model.trainable_variables):
                            if len(omegas_old) != 0:
                                omegas.append(1 / count_updates * gradients[pindex] + (1 - 1 / count_updates) *
                                              omegas_old[pindex])
                            else:
                                omegas.append(gradients[pindex])
                            star_variables.append(p.numpy())

                    else:
                        update_tags.append(0)
                    loss_window_means.append(loss_window_mean)
                    loss_window_variances.append(loss_window_variance)

                    # Hard Buffer Management
                    if use_hard_buffer:
                        loss = recent_loss.numpy()
                        new_samples = list(zip(loss.tolist(), x, y))

                        if self.history_buffer:  # Maintain historical buffer if enabled
                            history_hard_buffer.extend(new_samples)
                            history_hard_buffer = sorted(history_hard_buffer, key=lambda f: f[0], reverse=True)[:self.history_buffer_size]

                            # Extract labels from the history buffer
                            history_buffer_labels = [entry[2] for entry in history_hard_buffer]
                        
                            # Extract top 100 diverse samples for final hard buffer
                            max_class_count = max(main_window_distribution.values())
                            class_count = len(main_window_distribution)
                            print("Max class count: ", max_class_count)
                            max_per_class = (2 * self.hard_buffer_size) // class_count
                            print("Max samples per class: ", max_per_class)
                            
                            final_hard_buffer = []
                            sampled_classes = Counter()

                            for loss_val, sample, label in history_hard_buffer:
                                if sampled_classes[label] < max_per_class:
                                    final_hard_buffer.append((loss_val, sample, label))
                                    sampled_classes[label] += 1
                                if len(final_hard_buffer) >= self.hard_buffer_size:  
                                    break
                                
                            updated_hard_buffer_distribution = Counter([entry[2] for entry in final_hard_buffer])

                        else:
                            hard_buffer.extend(new_samples)
                            hard_buffer = sorted(hard_buffer, key=lambda f: f[0], reverse=True)[:self.hard_buffer_size]

                            final_hard_buffer = hard_buffer

                            updated_hard_buffer_distribution = Counter([entry[2] for entry in final_hard_buffer])

                        distribution.append({
                            "main_window_distribution": main_window_distribution,
                            "updated_hard_buffer_distribution": updated_hard_buffer_distribution
                        })

                    recent_buffer = []
                    update_times += 1
                    print(f"{update_times}th updating done")
                    print("------------------")

            prediction_results['actual_labels'] = actual_labels
            prediction_results['prediction_labels'] = prediction_labels

        return future_losses, prediction_results, distribution
