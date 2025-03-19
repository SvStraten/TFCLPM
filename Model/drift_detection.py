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
import numpy as np
import tensorflow as tf
from collections import Counter
import copy
from PrefixTreeCDD.PrefixTreeClass import PrefixTree
import PrefixTreeCDD.settings as settings
from PrefixTreeCDD.CDD import Window
from river import drift
import math
import edbn.Predictions.setting as setting
from numpy import log as ln
from collections import OrderedDict
from collections import deque
from itertools import islice
from scipy.stats import percentileofscore
from pm4py.streaming.importer.xes import importer as stream_xes_importer
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.conversion.log import converter as stream_converter
from pm4py.objects.log.exporter.xes import exporter as xes_exporter
import pm4py
from copy import deepcopy

def drift_detection(self, file):

    self.file = file
    # Read CSV
    xes_output = "filePath.xes"
    decay_lambda = 0.25
    noise = 0.8
    tree_size = 100

    endEventsDic = dict()
    window = Window(initWinSize=10)
    df = pd.read_csv(self.file)
    # print(df.columns)
    # Ensure timestamp is in datetime format
    df['completeTime'] = pd.to_datetime(df['completeTime'])

   
    # Convert DataFrame to an event log
    event_log = pm4py.format_dataframe(df, case_id='case', activity_key='event', timestamp_key='completeTime')
    
    # Export to XES
    pm4py.write_xes(event_log, xes_output)
    filePath = xes_output
    
    streaming_ev_object = stream_xes_importer.apply(os.path.abspath(filePath),
                                                    variant=stream_xes_importer.Variants.XES_TRACE_STREAM)

    # Process the log Trace-by-Trace
    for trace in streaming_ev_object:
        lastEvent = trace[-1]["event"]
        timeStamp = trace[-1]["completeTime"]
        caseID = trace.attributes["concept:name"]
        endEventsDic[caseID] = [lastEvent, timeStamp]
        
    caseList = [] # Complete list of cases seen
    Dcase = OrderedDict() # Dictionary of cases that we're tracking.

    tree = PrefixTree(pruningSteps = tree_size, noiseFilter=noise, lambdaDecay=decay_lambda) # Create the prefix tree with the first main node empty
    adwin = drift.ADWIN()
    ph = drift.PageHinkley()

    pruningCounter = 0 # Counter to check if pruning needs to be done
    traceCounter = 0  # Counter to create the Heuristics Miner model

    variant = xes_importer.Variants.ITERPARSE
    parameters = {variant.value.Parameters.TIMESTAMP_SORT: True, variant.value.Parameters.TIMESTAMP_KEY: 'completeTime'}
    log = xes_importer.apply(os.path.abspath(filePath),
                            variant=variant)  # , parameters=parameters)

    static_event_stream = stream_converter.apply(log, variant=stream_converter.Variants.TO_EVENT_STREAM, parameters=parameters)
    static_event_stream._list.sort(key=lambda x: x['completeTime'], reverse=False)

    eventCounter = 0 # Counter for number of events
    currentNode = tree.root  # Start from the root node

    start_time = time.time()
    drifts = []
    for ev in static_event_stream:
        caseList, Dcase, currentNode, pruningCounter, traceCounter, window = tree.insertByEvent(caseList, Dcase, currentNode, ev, pruningCounter, traceCounter, endEventsDic, window)
        eventCounter += 1

        if window.cddFlag:  # If a complete new tree has been created
            if len(window.prefixTreeList) == window.WinSize:  # Maximum size of window reached, start concept drift detection within the window
                temp_drifts = window.conceptDriftDetection(adwin, ph, eventCounter)
                window.WinSize = min(window.WinSize + 1, window.maxWindowSize)
                for i in temp_drifts:
                    if i not in drifts:
                        drifts.append(i)
                if len(window.prefixTreeList) == window.WinSize:  # If there was no drift detected within the window
                    window.prefixTreeList = deque(islice(window.prefixTreeList, 1, None))  # Drop the oldest tree

    end_time = time.time()
    print(end_time - start_time)
    print(drifts)

    return drifts