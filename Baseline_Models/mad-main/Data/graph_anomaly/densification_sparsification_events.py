import random
import csv
import argparse
import numpy as np
import injection
import utils

if __name__ == '__main__':

    # Parse commands
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--percentage_events', type=float)
    parser.add_argument('--event_size', type=int)
    parser.add_argument('--seed', type=int, default = 42)
    args = parser.parse_args()

    # Set random seed
    random.seed(args.seed)

    # Set file paths
    data_path = '../clean_data/' + args.dataset + '.txt'
    data_out_path = './anomalous_data/' + args.dataset + '_densif_sparsif_data.txt'
    label_path = './anomalous_data/' + args.dataset + '_densif_sparsif_gt.txt'

    # Read clean data
    data = utils.read_data(data_path)
    t_max = data[-1][2]

    # As timeseries of graphs
    data_tsg = utils.time_series_graphs(data)

    # Compute the anomalies
    inject_anomalies = injection.anomalies_densification(data_tsg, args.percentage_events, args.event_size)
    remove_anomalies = injection.anomalies_sparsification(data_tsg, args.percentage_events, args.event_size)

    # Apply the anomalies 
    for anom in inject_anomalies: data.append( anom )
    for anom in remove_anomalies: data.remove( anom )
    data = sorted( data, key=lambda x:x[2] )
    utils.save_data(data_out_path, data)

    # Add labels and store them. 
    inj_anom_times = { anom[2] for anom in inject_anomalies }
    rem_anom_times = { anom[2] for anom in remove_anomalies }
    anom_times = inj_anom_times.union( rem_anom_times )
    out_labels = [(t, 0) for t in range(t_max+1) if t not in anom_times]
    out_labels += [(t_anom, 1) for t_anom in anom_times]
    utils.save_data(label_path, out_labels)

    # Print States
    print('--- Densification and Sparsification Done. Processed ' + args.dataset + ' dataset ---')
    print('Num of anomalous timestamps: ', len(anom_times))
    print('Num anomalous triplets for densification: ', len(inject_anomalies))
    print('Num anomalous triplets for sparsification: ', len(remove_anomalies))
