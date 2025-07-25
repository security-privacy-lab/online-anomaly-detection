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
    parser.add_argument('--event_size', type=int)
    parser.add_argument('--seed', type=int, default = 42)
    args = parser.parse_args()

    # Set random seed
    random.seed(args.seed)

    # Set file paths
    data_path = '../clean_data/' + args.dataset + '.txt'
    data_out_path = './anomalous_data/' + args.dataset + '_rewiring_data.txt'
    label_path = './anomalous_data/' + args.dataset + '_rewiring_gt.txt'
    queries_path = './anomalous_data/' + args.dataset + '_rewiring_queries.txt'

    # Read clean data
    data = utils.read_data(data_path)
    t_max = data[-1][2]

    # As temporal nodes adjacencies
    data_tna = utils.temporal_node_adj(data)

    # Get query nodes
    origin_node_set = list(data_tna.keys())
    query_nodes = random.sample( origin_node_set, args.num_queries )  

    # Compute the anomalies
    inject_anomalies = []
    remove_anomalies = []
    to_clean = []
    for q_n in query_nodes: 
        to_inject, to_remove = injection.anomalies_rewiring(q_n, data_tna, args.percentage_events, args.event_size)
        if len(to_inject) > 0 : inject_anomalies += to_inject
        if len(to_remove) > 0 : remove_anomalies += to_remove
        if len(to_inject) == 0 and len(to_remove) == 0: to_clean.append( q_n )

    # Remove query nodes with no anomalies
    query_nodes = [q_n for q_n in query_nodes if q_n not in to_clean]

    # Apply the rewiring 
    for anom in inject_anomalies: data.append( anom )
    for anom in remove_anomalies: data.remove( anom )
    data = sorted( data, key=lambda x:x[2] )
    utils.save_data(data_out_path, data)

    # Add labels and store them. Inject and remove times are equal
    anom_times = { q_n : set() for q_n in query_nodes }
    for anom in inject_anomalies : anom_times[anom[0]].add( anom[2] ) 
    out_labels = [(q_n, t, 0) for q_n in query_nodes for t in range(t_max+1) if t not in anom_times[q_n]]
    out_labels += list( {(anom[0], anom[2], 1) for anom in inject_anomalies} )
    utils.save_data(label_path, out_labels)

    # Store queries 
    query_nodes = [[q_n] for q_n in query_nodes]
    utils.save_data(queries_path, query_nodes)

    # Print States
    print('--- Rewiring Done. Processed ' + args.dataset + ' dataset ---')
    print('Num query nodes: ', len(query_nodes))
    print('Num of rewired triplets: ', 2*len(inject_anomalies))
