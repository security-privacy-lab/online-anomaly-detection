from matplotlib import gridspec
import matplotlib as mpl
import matplotlib.pyplot as plt
import MAD.model as model
import gdecomp as gdec
import gdecomp.dictionary as gdic
import gdecomp.decomposition as gtr
import numpy as np
import random

def plot_spectrum(spec, axis, scale, color):	
    levels = int(np.ceil(np.log2(len(spec))))
    # levels of resolution
    bins = [2**(i) for i in range(0, levels)]
    # plot figure
    axis.vlines(x=bins, ymin=0, ymax=scale, colors='gray', ls='--', lw=1, label='levels')
    axis.plot(np.arange(1, len(spec)+1), spec, color=color)
    axis.set_xscale('log')
    #axis.spines['right'].set_visible(False)
    #axis.spines['top'].set_visible(False)
    print('done')
    return axis


def anom_scores(Q, P, Tree):
    # Query to random variables
    x = np.hstack( gtr.decomposition_coefficients(Q, Tree) )

    # Get theoretical mean and variances
    th_mean, th_var = model.get_model_mean_var(P, Tree)  

    # Compute anomaly score
    score = []
    for i in range(len(x)):
        score += [ (x[i] - th_mean[i])**2 / max(th_var[i], 1e-2) ]
    return score


######################
###### SET MODEL #####
######################

# Number of blocks
num_blocks = 2
size_blocks = 32
N = num_blocks*size_blocks

# Define the blocks
C = [[l*size_blocks + k for k in range(size_blocks)] for l in range(num_blocks)]

# Define the edge probabilities
pin = 0.8
pout = 0.5

# draw graph realizations at random
P = gdec.Graph() 
phi = []
# Process links between community i and community j
for c_i in range(len(C)):
    for c_j in range(len(C)):
        if c_i == c_j:
            for idx_u, u in enumerate(C[c_i]):
                for idx_v, v in enumerate(C[c_j][::-1]):
                    phi.append( (u,v) )
                    P.add_edge( (u,v) )
                    P.set_edge_weight( (u,v), pin**(idx_u + idx_v + 2) )
        else:
            for idx_u, u in enumerate(C[c_i]):
                for idx_v, v in enumerate(C[c_j][::-1]):
                    phi.append( (u,v) )
                    P.add_edge( (u,v) )
                    P.set_edge_weight( (u,v), pout**(idx_u + idx_v + 2) )

########################
###### DICTIONARY ######
########################
Tree = gdic.DictionarySubgraph(phi)
Tree.order_rank(P.to_weightfn())

########################
#### NORMAL GRAPH ######
########################
normal_graph = [edge for edge,p in P.to_weightfn().items() if np.random.rand() < p]
query_normal_graph = gdec.Graph.from_weightfn({e:1 if e in normal_graph else 0 for e in phi})

# Anomaly scores
score_normal = anom_scores(query_normal_graph, P, Tree)

########################
#### DENSIF GRAPH ######
########################
new_edges = random.sample(list(set(phi).difference(normal_graph)), len(normal_graph))
densif_graph = normal_graph + new_edges
query_densif_graph = gdec.Graph.from_weightfn({e:1 if e in densif_graph else 0 for e in phi})

# Anomaly scores
score_densif = anom_scores(query_densif_graph, P, Tree)

########################
#### SPARSIF GRAPH #####
########################
sparsif_graph = []
query_sparsif_graph = gdec.Graph.from_weightfn({e:1 if e in sparsif_graph else 0 for e in phi})

# Anomaly scores
score_sparsif = anom_scores(query_sparsif_graph, P, Tree)

########################
#### REWIRE GRAPH ######
########################
query_rewired_graph = gdec.Graph.from_weightfn({e:1 if e in new_edges else 0 for e in phi})
# Anomaly scores
score_rewired = anom_scores(query_rewired_graph, P, Tree)


########################
####### FIGURES ########
########################
fig = plt.figure(figsize=[7,3])
gs = gridspec.GridSpec(4, 1)

ax0 = plt.subplot(gs[0])
ax0 = plot_spectrum(score_normal, ax0, 80, 'black')
ax0.yaxis.set_tick_params(labelsize=10)
plt.xscale('log')
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
textstr = 'Normal'
ax0.text(0.05, 0.85, textstr, transform=ax0.transAxes, fontsize=10, verticalalignment='top', bbox=props)
plt.title('Scores for individual random variables', fontweight="bold")

ax1 = plt.subplot(gs[1])
ax1 = plot_spectrum(score_densif, ax1, 80, 'magenta')
ax1.yaxis.set_tick_params(labelsize=10)
plt.setp(ax0.get_xticklabels(), visible=False)
plt.subplots_adjust(hspace=.2)
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
textstr = 'Densification'
ax1.text(0.05, 0.85, textstr, transform=ax1.transAxes, fontsize=10, verticalalignment='top', bbox=props)
plt.xscale('log')

ax2 = plt.subplot(gs[2])
ax2 = plot_spectrum(score_sparsif, ax2, 80, 'blue')
ax2.yaxis.set_tick_params(labelsize=10)
plt.setp(ax1.get_xticklabels(), visible=False)
plt.subplots_adjust(hspace=.05)
plt.xscale('log')
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
textstr = 'Sparsification'
ax2.text(0.05, 0.85, textstr, transform=ax2.transAxes, fontsize=10, verticalalignment='top', bbox=props)

ax3 = plt.subplot(gs[3])
ax3 = plot_spectrum(score_rewired, ax3, 80, 'red')
ax3.yaxis.set_tick_params(labelsize=10)
plt.setp(ax2.get_xticklabels(), visible=False)
plt.subplots_adjust(hspace=.05)
plt.xscale('log')
plt.xlabel('$i$', fontsize=13)
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
textstr = 'Rewiring'
ax3.text(0.05, 0.85, textstr, transform=ax3.transAxes, fontsize=10, verticalalignment='top', bbox=props)

plt.savefig('signature', dpi=300, edgecolor='w',bbox_inches="tight")
