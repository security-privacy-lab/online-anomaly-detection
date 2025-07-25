import MAD.MAD
import utils
import argparse
import lsmanip
import lsmanip.io as lio
import lsmanip.subset as lsub

if __name__ == '__main__':

    # Parse commands
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--window_size', type=int)
    parser.add_argument('--only_active', action="store_true")
    parser.add_argument('--time_init', type=bool, default=10)
    args = parser.parse_args()
    
    # Set file paths
    data_path = './datasets/' + args.dataset + '_data.txt'
    queries_path = './queries/' + args.dataset + '_queries.txt'
    result_path = './result/' + args.dataset + '_scores.txt'
    print('Processing ' + args.dataset + ' dataset ...')

    # Load data
    link_stream = lio.from_csv(data_path)

    # Load queries 
    query_list = utils.read_queries(queries_path)

    # Process each query over time
    out_list = []
    for query in query_list:
        # construct phi 
        phi = [query]

        # Get the activity of phi through time
        ls_phi = lsub.subgraph_partition(link_stream,phi)
        ls_phi_tsg = ls_phi.to_tsgraphs()

        # Analyze phi through time
        for t in range(args.time_init, link_stream.max_time + 1):
            
            # If only active edges is set
            if (args.only_active == True) and (t not in ls_phi_tsg): continue

            # Set query
            if t in ls_phi_tsg:
                Q = {(u,v,t):1 if (u,v) in ls_phi_tsg[t] else 0 for u,v in phi}
            else:
                Q = {(u,v,t):0 for u,v in phi}

            # Compute context 
            H = lsub.time_partition(ls_phi, [t - args.window_size, t]).to_triplets()

            # Score the query
            score = MAD.MAD.scoring(Q, H)

            # Store result
            out_list += [(u,v,t,score) for u,v in phi] 

    utils.save_data(result_path, out_list)
    print('Done')
