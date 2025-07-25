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
    parser.add_argument('--num_queries', type=int)
    parser.add_argument('--percentage_events', type=float)
    parser.add_argument('--seed', type=int, default = 42)
    args = parser.parse_args()

    # Set random seed
    random.seed(args.seed)

    # Set file paths
    data_path = '../clean_data/' + args.dataset + '.txt'
    data_out_path = './anomalous_data/' + args.dataset + '_sparsif_data.txt'
    label_path = './anomalous_data/' + args.dataset + '_sparsif_gt.txt'
    queries_path = './anomalous_data/' + args.dataset + '_sparsif_queries.txt'

    # Read clean data
    data = utils.read_data(data_path)

    # As edge-time series
    data_ets = utils.edge_timets(data)

    # Get query edges
    relation_set = list(data_ets.keys())
    query_edges = random.sample( relation_set, args.num_queries )  

    # Compute anomalous edges
    anomalies = []
    to_clean = []
    for q_e in query_edges: 
        to_remove = injection.anomalies_sparsification(q_e, data_ets, args.percentage_events)
        if len(to_remove) > 0 : anomalies += to_remove
        else: to_clean.append( q_e )

    # Remove query edges with no anomalies injected
    query_edges = [q_e for q_e in query_edges if q_e not in to_clean]

    # Inject the anomalies (remove the edges)
    for anom in anomalies: data.remove( anom )

    # Save the anomalous dataset
    utils.save_data(data_out_path, data)

    # Store queries and their labels
    t_max = data[-1][2]
    anom_times = {q_e : set() for q_e in query_edges}
    for anom in anomalies: anom_times[(anom[0],anom[1])].add( anom[2] )
    out_labels = [(q_e[0],q_e[1],t,0) for q_e in query_edges for t in range(t_max + 1) if t not in anom_times[q_e]]
    out_labels += [(anom[0], anom[1], anom[2], 1) for anom in anomalies]
    utils.save_data(label_path, out_labels)

    # Store queries 
    utils.save_data(queries_path, query_edges)

    # Print States
    print('--- Sparsification Done. Processed ' + args.dataset + ' dataset ---')
    print('Num query edges: ', len(query_edges))
    print('Num anomalous edges added: ', len(anomalies))
