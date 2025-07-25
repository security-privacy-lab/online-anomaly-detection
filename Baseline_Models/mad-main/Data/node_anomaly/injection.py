import numpy as np
import random

def anomalies_densification(query, data_tna, percentage, event_size):
    # Get number of event times where anomalies are injected
    activity_times = list(data_tna[query].keys())
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

        # Inject edges towards vertices already visited
        neighbors = []
        if abnormal_time in data_tna[query]: neighbors = data_tna[query][abnormal_time]
        past_neighbors = {v for t in data_tna[query] for v in data_tna[query][t] if t < abnormal_time}
        neighbors_pool = list(past_neighbors.difference(set(neighbors)))

        true_event_size = min([len(neighbors_pool), event_size])
        anom_v = random.sample(neighbors_pool, true_event_size)

        to_inject += [ (query, v, abnormal_time) for v in anom_v ]

    return to_inject

def anomalies_sparsification(query, data_tna, percentage, event_size):
    # Get number of event times where anomalies are injected
    activity_times = list(data_tna[query].keys())
    num_events = int(len(activity_times)*percentage/100)
    ts_activity = [len(data_tna[query][t]) for t in data_tna[query]]
    to_remove = []

    for event in range(num_events):

        # Pick instant of largest activity
        idx_peak = np.argmax(ts_activity)
        abnormal_time = activity_times[ idx_peak ]
        ts_activity[idx_peak] = 0

        # After setting times of removal, we set the edges to remove
        neighbors = data_tna[query][abnormal_time]
        if len(neighbors) > event_size:
            anom_v = random.sample(neighbors, event_size)
        else: anom_v = neighbors
        to_remove += [(query, v, abnormal_time) for v in anom_v]

    return to_remove

def anomalies_rewiring(query, data_tna, percentage, event_size):
    # Get number of event times where anomalies are injected
    activity_times = list(data_tna[query].keys())
    num_events = int(len(activity_times)*percentage/100)
    to_inject = []
    to_remove = []

    for event in range(num_events):
        
        # Pick a snapshot at random
        abnormal_time = random.sample(activity_times, 1)[0]
        activity_times.remove(abnormal_time)

        # Get the neighbors and past neighrbors
        neighbors = data_tna[query][abnormal_time]
        past_neighbors = {v for t in data_tna[query] for v in data_tna[query][t] if t < abnormal_time}
        neighbors_pool = list(past_neighbors.difference(set(neighbors)))

        # Injection and removal 
        true_event_size = min([len(neighbors), len(neighbors_pool), event_size])
        rem_rewiring_set = random.sample(neighbors, true_event_size) 
        inj_rewiring_set = random.sample(neighbors_pool, true_event_size)
        to_remove += [(query, v, abnormal_time) for v in rem_rewiring_set]
        to_inject += [(query, v, abnormal_time) for v in inj_rewiring_set]

    return to_inject, to_remove
