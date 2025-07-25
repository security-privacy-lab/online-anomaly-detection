import numpy as np
import random

def anomalies_densification(data_tsg, percentage, event_size):
    # Get number of event times where anomalies are injected
    activity_times = list(data_tsg.keys())
    num_events = int(len(activity_times)*percentage/100)
    to_inject = []

    for event in range(num_events):

        # Inject anomaly in the middle of largest gap of inactivity. If no gap then pick randomly.
        time_spacing = np.diff( activity_times )
        if np.max(time_spacing) == 1: 
            slot = random.randint(0, len(time_spacing))
            abnormal_time = activity_times[slot]
        else: 
            slot = np.argmax(time_spacing)
            abnormal_time = activity_times[slot] + time_spacing[slot]//2
            activity_times = sorted( activity_times + [abnormal_time] )

        # Inject edges that we have already seen
        curr_graph = []
        if abnormal_time in data_tsg: curr_graph = data_tsg[abnormal_time]
        past_edges = {e for t in data_tsg for e in data_tsg[t] if t < abnormal_time}
        edge_pool = list(past_edges.difference(set(curr_graph)))

        true_event_size = min([len(edge_pool), event_size])
        anom_edges = random.sample(edge_pool, true_event_size)

        to_inject += [ (e[0], e[1], abnormal_time) for e in anom_edges ]

    return to_inject

def anomalies_sparsification(data_tsg, percentage, event_size):
    # Get number of event times where anomalies are injected
    activity_times = list(data_tsg.keys())
    num_events = int(len(activity_times)*percentage/100)
    ts_activity = [len(data_tsg[t]) for t in data_tsg]
    to_remove = []

    for event in range(num_events):

        # Pick instant of largest activity
        idx_peak = np.argmax(ts_activity)
        abnormal_time = activity_times[ idx_peak ]
        ts_activity[idx_peak] = 0

        # After setting times of removal, we set the edges to remove
        graph = data_tsg[abnormal_time]
        if len(graph) > event_size:
            anom_edges = random.sample(graph, event_size)
        else: anom_edges = graph
        to_remove += [(anom[0], anom[1], abnormal_time) for anom in anom_edges]

    return to_remove

def anomalies_rewiring(data_tsg, percentage, event_size):
    # Get number of event times where anomalies are injected
    activity_times = list(data_tsg.keys())
    num_events = int(len(activity_times)*percentage/100)
    to_inject = []
    to_remove = []

    for event in range(num_events):
        
        # Pick a snapshot at random
        abnormal_time = random.sample(activity_times, 1)[0]
        activity_times.remove(abnormal_time)

        # Get the current graph and past seen edges
        curr_graph = data_tsg[abnormal_time]
        past_edges = {e for t in data_tsg for e in data_tsg[t] if t < abnormal_time}
        edge_pool = list(past_edges.difference(set(curr_graph)))

        # Inject edges that we have only seen in the past
        true_event_size = min([len(curr_graph), len(edge_pool), event_size])
        rem_rewiring_set = random.sample(curr_graph, true_event_size) 
        inj_rewiring_set = random.sample(edge_pool, true_event_size)
        to_remove += [(e[0], e[1], abnormal_time) for e in rem_rewiring_set]
        to_inject += [(e[0], e[1], abnormal_time) for e in inj_rewiring_set]

    return to_inject, to_remove
