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
    data_out_path = './anomalous_data/' + args.dataset + '_densif_data.txt'
    label_path = './anomalous_data/' + args.dataset + '_densif_gt.txt'
    queries_path = './anomalous_data/' + args.dataset + '_densif_queries.txt'

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
        to_inject = injection.anomalies_densification(q_e, data_ets, args.percentage_events)
        if len(to_inject) > 0 : anomalies += to_inject
        else: to_clean.append( q_e )

    # Remove query edges with no anomalies injected
    query_edges = [q_e for q_e in query_edges if q_e not in to_clean]

    # Inject the anomalies into the data
    for anom in anomalies: data.append( anom )
    data = sorted( data, key=lambda x:x[2] )

    # Save the anomalous dataset
    utils.save_data(data_out_path, data)

    # Store queries and their labels
    out_labels = [(q_e[0],q_e[1],t,0) for q_e in query_edges for t in data_ets[q_e]]
    out_labels += [(anom[0], anom[1], anom[2], 1) for anom in anomalies]
    utils.save_data(label_path, out_labels)

    # Store queries 
    utils.save_data(queries_path, query_edges)

    # Print States
    print('--- Densification Done. Processed ' + args.dataset + ' dataset ---')
    print('Num query edges: ', len(query_edges))
    print('Num anomalous edges added: ', len(anomalies))
