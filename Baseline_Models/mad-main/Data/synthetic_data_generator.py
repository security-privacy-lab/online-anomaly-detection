import csv 
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# Nnumber of times
num_times = 5000

# Number of blocks
num_blocks = 10
size_blocks = 500
N = num_blocks*size_blocks

# Define the blocks
C = [[l*size_blocks + k for k in range(size_blocks)] for l in range(num_blocks)]

# Define the edge probabilities
pin = 0.8
pout = 0.5

exp_in = sum( [pin**(u + v) for u in range(1, size_blocks + 1) for v in range(1, size_blocks + 1)] )
exp_out = sum( [pout**(u + v) for u in range(1, size_blocks + 1) for v in range(1, size_blocks + 1)] )

print('Expected number of edges within communities : ', exp_in)
print('Expected number of edges between communities : ', exp_out)

# draw graph realizations at random
edgelist = []
for t in range(num_times):
    # Process links between community i and community j
    for c_i in range(len(C)):
        for c_j in range(len(C)):
            if c_i == c_j:
                for idx_u, u in enumerate(C[c_i]):
                    for idx_v, v in enumerate(C[c_j][::-1]):
                        prob = pin**(idx_u + idx_v + 2)
                        if np.random.rand() < prob:
                            edgelist.append( (u, v, t) )
                        if prob < 1e-6:
                            break
            else:
                for idx_u, u in enumerate(C[c_i]):
                    for idx_v, v in enumerate(C[c_j][::-1]):
                        prob = pout**(idx_u + idx_v + 2)
                        if np.random.rand() < prob:
                            edgelist.append( (u, v, t) )
                        if prob < 1e-6:
                            break
nodes = set()
for triplet in edgelist: nodes.add(triplet[0]); nodes.add(triplet[1]);
print('number of nodes = ', len(nodes))

with open('clean_data/synthetic.txt','w') as out:
	csv_out=csv.writer(out, delimiter=',')
	for triplet in edgelist: csv_out.writerow(triplet)

print('done')
