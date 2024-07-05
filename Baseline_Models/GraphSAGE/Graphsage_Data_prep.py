import json
import numpy as np
import networkx as nx
from networkx.readwrite import json_graph

with open('Baseline_Models/graph_creation/processed_data/max_pygraphs-G.json') as f:
    graph_json = json.load(f)

with open('Baseline_Models/graph_creation/processed_data/max_pygraphs-id_map.json') as f:
    id_map = json.load(f)

with open('Baseline_Models/graph_creation/processed_data/max_pygraphs-class_map.json') as f:
    class_map = json.load(f)

G = json_graph.node_link_graph(graph_json)

G = nx.relabel_nodes(G, str)

# Print attributes of the first few nodes to check features
for node in list(G.nodes())[:5]:
    print(G.nodes[node])


features = np.array([G.nodes[node]['features'] for node in G.nodes()])
labels = np.array([class_map[str(node)] for node in G.nodes()])

np.save('features.npy', features)
np.save('labels.npy', labels)

nx.write_adjlist(G, 'graph.adj')