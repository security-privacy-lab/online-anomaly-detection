import MAD.gdecomp as gdec
import MAD.gdecomp.dictionary as gdic
import MAD.gdecomp.decomposition as gtr
import MAD.lsmanip as lsm
import MAD.lsmanip.subset as lsub
import MAD.lsmanip.aggregation as lagg
import numpy as np

def fit(phi, H):

    # Model is a weighted graph P : phi -> [0,1]
    P = gdec.Graph.from_weightfn( {e:0 for e in phi} )
    if H.size == 0: return P

    # Make stationarity test for windows of all sizes
    best_fit = None
    for k in range(1, H.max_time - H.min_time):
        H_window = lsub.time_partition(H, [H.max_time - k, H.max_time + 1])
        if H_window.size == 0: continue

        H_window.max_time = H.max_time
        H_window.min_time = H.max_time - k

        fitness_score = stationarity_test(phi, H_window)

        # Retain the best model for the window
        if best_fit == None: best_fit = fitness_score
        if fitness_score <= best_fit :
            best_fit = fitness_score
            P = get_model(phi, H_window)

    return P


def stationarity_test(phi, H_window):

    # Trivial case
    if H_window.size == 0: return 0

    # Estimate model for window
    P = get_model(phi, H_window)

    # Set the tree (dictionary) to perform decomposition
    Tree = gdic.DictionarySubgraph(phi)
    Tree.order_rank(P.to_weightfn())

    # Decompose slices
    tsg = H_window.to_tsgraphs()
    slices_to_decompose = [gdec.Graph.from_weightfn(tsg[t]) for t in tsg]
    decomposed_slices = [gtr.decomposition_coefficients(g, Tree) for g in slices_to_decompose]
    decomposed_slices = [np.hstack(x) for x in decomposed_slices]

    # Sample mean
    delta = H_window.max_time - H_window.min_time + 1
    X = np.vstack( decomposed_slices )
    sample_mean = np.sum(X, axis=0) / delta

    # Sample variance
    sq_mean = (delta - len(decomposed_slices))*np.power(sample_mean, 2)
    X_shift = np.vstack([x - sample_mean for x in decomposed_slices])
    sample_var = (np.sum(np.power(X_shift, 2), axis=0) + sq_mean)/delta
    
    # Theoretical variance    
    th_mean, th_var = get_model_mean_var(P, Tree)

    # Fitness score
    fitness = np.linalg.norm(th_var - sample_var, 2)
    
    return fitness

def get_model(phi, H):
    count = lagg.aggregated_graph( H )
    delta = H.max_time - H.min_time + 1
    P = gdec.Graph.from_weightfn({e:count[e]/delta if e in count else 0 for e in phi})
    return P

def get_model_mean_var(P, Tree):

    # Mean
    th_mean = np.hstack(gtr.decomposition_coefficients(P, Tree))

    # Variance
    var_model = gdec.Graph.from_weightfn({e:p*(1-p) for e,p in P.to_weightfn().items()}) 

    # Use the tree to compute the variance values
    var_dec = gtr.decomposition_coefficients(var_model, Tree)
    var_dec = gtr.multiscale_approximation(var_dec)

    # It remains to normalize
    tree_leaves_size = len(var_dec[-1])
    for level in range(len(var_dec)): 
        var_dec[level] = np.divide(var_dec[level], np.sqrt(tree_leaves_size/(2**level)))

    th_var = np.hstack([var_dec[0]] + var_dec[0:-1])

    return th_mean, th_var
