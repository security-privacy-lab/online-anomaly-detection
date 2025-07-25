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
    parser.add_argument('--time_init', type=int, default = 10)
    args = parser.parse_args()

    # Set file paths
    data_path = './datasets/' + args.dataset + '_data.txt'
    result_path = './result/' + args.dataset + '_scores.txt'
    print('Processing ' + args.dataset + ' dataset ...')

    # Load data
    link_stream = lio.from_csv(data_path)

    # Construct phi. 
    """ Construction of phi.
    Irrelevant to track edges that we know never appear and always have 0 probability. 
    This allows to obtain a significant speed-up by reducing the size of phi. 
    
    In an offline scenario it is not serious to exclude such edges:
    they always appear in the right-part of the tree and never have any activity. 

    In a real-time tracking scenario it may be better to include all the edge space in phi 
    to account for edges that we have never seen but may appear for the first time. """
    phi = list({(u,v) for u,v,t in link_stream.to_triplets()})
    
    # Link stream as tsgraphs
    ls_tsg = link_stream.to_tsgraphs()

    # Process the each timestamp
    out_list = []
    for t in range(args.time_init, link_stream.max_time + 1):
        if t % 100 == 0:
            print('Processing timestamp ' + str(t) + ' ...')

        # Set query
        if t in ls_tsg:
            Q = {(u,v,t):1 if (u,v) in ls_tsg[t] else 0 for u,v in phi}
        else:
            Q = {(u,v,t):0 for u,v in phi}

        # Compute context 
        H = lsub.time_partition(link_stream, [t - args.window_size, t]).to_triplets()
        
        # Score the query
        score = MAD.MAD.scoring(Q, H)

        # Store result
        out_list += [(t, score)]

    utils.save_data(result_path, out_list)
    print('Done')
