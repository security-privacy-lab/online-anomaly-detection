import numpy as np

def anomalies_densification(query, data_ets, percentage):
    # get number of anomalous links
    activity_times = data_ets[query].copy()
    num_events = int(len(activity_times)*percentage/100)
    to_inject = []
    
    # Inject the anomalies between the largest activity gaps
    for event in range(num_events):
        time_spacing = np.diff( activity_times )
        slot = np.argmax(time_spacing)
        abnormal_time = activity_times[slot] + time_spacing[slot]//2
        activity_times = sorted( activity_times + [abnormal_time] )
        to_inject.append( (query[0], query[1], abnormal_time) )
    return to_inject

def anomalies_sparsification(query, data_ets, percentage):
    # get the number of anomalous links to remove
    activity_times = data_ets[query].copy()
    num_events = int(len(activity_times)*percentage/100)
    to_remove = []

    # Remove the edges that have the least activity gaps
    for event in range(num_events):
        time_spacing = np.diff( activity_times )
        slot = np.argmin(time_spacing)
        abnormal_time = activity_times[slot + 1]
        activity_times.remove(abnormal_time)
        to_remove.append( (query[0], query[1], abnormal_time) )

    return to_remove
