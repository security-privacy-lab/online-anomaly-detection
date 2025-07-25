import MAD.model as model
import MAD.gdecomp as gdec
import MAD.gdecomp.dictionary as gdic
import MAD.gdecomp.decomposition as gtr
import MAD.lsmanip as lsm
import MAD.lsmanip.aggregation as lagg
import MAD.lsmanip.subset as lsub
import numpy as np

# Query is a dict[ (u,v,t) ] -> {0,1} and t is fixed
# Context is L( t-K : t-1, phi ) 
def scoring(query, context):

    # Prepare data
    Q_timestamped = lsm.LinkStream.from_weightfn(query)
    Q_timestamp = Q_timestamped.max_time 
    Q = gdec.Graph.from_weightfn( lagg.aggregated_graph(Q_timestamped) )
    phi = Q.to_edgelist()

    H = lsm.LinkStream.from_triplets(context)
    H = lsub.time_partition(H, [H.min_time, Q_timestamp])
    H = lsub.subgraph_partition( H, phi )
    H.max_time = Q_timestamp - 1

    # Compute the model
    P = model.fit(phi, H)

    # Use the model to set the tree (sorted by prob order)
    Tree = gdic.DictionarySubgraph(phi)
    Tree.order_rank(P.to_weightfn())

    # Query to random variables
    x = np.hstack( gtr.decomposition_coefficients(Q, Tree) )

    # Get theoretical mean and variances
    th_mean, th_var = model.get_model_mean_var(P, Tree)  

    # Compute anomaly score
    score = 0
    for i in range(len(x)):
        score += (x[i] - th_mean[i])**2 / max(th_var[i], 1e-2)
    
    return score
