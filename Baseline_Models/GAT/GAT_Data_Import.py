import json
import torch
from torch_geometric.data import Data
import numpy as np

# Load the graph data, ID map, and class map
with open("graph_creation/processed_data/gat_pygraphs.json", "r") as f:
    graph_data = json.load(f)

with open("graph_creation/processed_data/gat_id_map.json", "r") as f:
    id_map = json.load(f)

with open("graph_creation/processed_data/gat_class_map.json", "r") as f:
    class_map = json.load(f)

# Ensure all IDs are strings
id_map = {str(k): v for k, v in id_map.items()}
class_map = {str(k): v for k, v in class_map.items()}

# Create edge index
edges = [(link["source"], link["target"]) for link in graph_data["links"]]
edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

num_nodes = len(graph_data["nodes"])
num_features = len(graph_data["nodes"][0]["features"])

x = torch.zeros((num_nodes, num_features), dtype=torch.float)
y = torch.zeros(num_nodes, dtype=torch.long)

missing_ids = []
for node in graph_data["nodes"]:
    node_id = str(node["id"])
    if node_id not in id_map:
        missing_ids.append(node_id)
        continue
    index = id_map[node_id]
    x[index] = torch.tensor(node["features"], dtype=torch.float)
    y[index] = torch.tensor(class_map[node_id], dtype=torch.long)

if missing_ids:
    print(f"Missing node IDs in id_map: {missing_ids}")

data = Data(x=x, edge_index=edge_index, y=y)

num_train = int(0.8 * num_nodes)
train_mask = torch.zeros(num_nodes, dtype=torch.bool)
test_mask = torch.zeros(num_nodes, dtype=torch.bool)

indices = np.random.permutation(num_nodes)
train_indices = indices[:num_train]
test_indices = indices[num_train:]

train_mask[train_indices] = True
test_mask[test_indices] = True

data.train_mask = train_mask
data.test_mask = test_mask

torch.save(data, "graph_creation/processed_data/gat_data.pt")
