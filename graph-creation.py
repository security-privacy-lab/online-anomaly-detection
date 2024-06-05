# First, the necessary modules are imported
import pandas
import torch
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx
import networkx as nx

df = pandas.read_csv("Darknet.CSV")
# We first properly convert the timestamps to panda's native format.
df["Timestamp"] = pandas.to_datetime(df["Timestamp"], format="%d/%m/%Y %I:%M:%S %p")
# Then create a rounded to the minute field for use later.
df["timestamp_minute"] = df["Timestamp"].dt.floor("min")
# we sort the dataframe by timestamp earliest to latest...
df = df.sort_values(by=["Timestamp"])
# and group the entries by minute...
grouped = df.groupby('timestamp_minute')

total_groups = len(grouped)
progress_counter = 0
pyg_graphs = []
for timestamp_minute, group in grouped:
    # For testing, nodes will be named by their index.
    index = 0
    # Then, we create the graph.
    G = nx.DiGraph()
    # And maintain the list of nodes.
    nodes_list = []
    # Lastly, we establish a previous node variable to store for easier graph creation.
    previous_backward_node = None
    # Now, for each entry...
    for idx, row in group.iterrows():
        # Record the source and destination IP, as well as the flow ID
        src_ip = row['Src IP']
        dst_ip = row['Dst IP']
        flow_id = row['Flow ID']
        
        # Initialize node features dictionary with common attributes for both forward and backward nodes
        node_features = {
            'src_ip': src_ip,
            'dst_ip': dst_ip,
            'flow_id': flow_id,
            'direction': 'bidirectional'  # Adding a direction attribute to distinguish between forward and backward nodes
        }
        
        # Update the dictionary with specific attributes for the forward node
        forward_features = {
            'total_pkts': row['Total Fwd Packet'],
            'pkts_per_s': row['Fwd Packets/s'],
            'iat_std': row['Fwd IAT Std'],
            'total_len_pkt': row['Total Length of Fwd Packet'],
            'pkt_len_std': row['Fwd Packet Length Std'],
            'seg_size_avg': row['Fwd Segment Size Avg'],
            'init_win_bytes': row['FWD Init Win Bytes'],
            'pkt_len_mean': row['Fwd Packet Length Mean'],
            'iat_max': row['Fwd IAT Max'],
            'avg_pkt_size': row['Average Packet Size'],
            'subflow_bytes': row['Subflow Fwd Bytes'],
            'ip': src_ip,
            'direction': 'forward'
        }
        node_features.update(forward_features)
        
        # Add the forward node to the graph
        forward_node_id = f"{index}_fwd"
        G.add_node(forward_node_id, **node_features)
        nodes_list.append(forward_node_id)
        
        # Update the dictionary with specific attributes for the backward node
        backward_features = {
            'total_pkts': row['Total Bwd packets'],
            'pkts_per_s': row['Bwd Packets/s'],
            'iat_std': row['Bwd IAT Std'],
            'total_len_pkt': row['Total Length of Bwd Packet'],
            'pkt_len_std': row['Bwd Packet Length Std'],
            'seg_size_avg': row['Bwd Segment Size Avg'],
            'init_win_bytes': row['Bwd Init Win Bytes'],
            'pkt_len_mean': row['Bwd Packet Length Mean'],
            'iat_max': row['Bwd IAT Max'],
            'avg_pkt_size': row['Average Packet Size'],
            'subflow_bytes': row['Subflow Bwd Bytes'],
            'ip': dst_ip,
            'direction': 'backward'
        }
        node_features.update(backward_features)
        
        # Add the backward node to the graph
        backward_node_id = f"{index}_bkd"
        G.add_node(backward_node_id, **node_features)
        nodes_list.append(backward_node_id)
        
        # Connect the edges as appropriate
        G.add_edge(forward_node_id, backward_node_id)
        if previous_backward_node is not None:
            G.add_edge(previous_backward_node, forward_node_id)
        
        previous_backward_node = backward_node_id
        index += 1

        # Convert the NetworkX graph to a PyTorch Geometric Data object
        data = from_networkx(G)

        # Append the PyG Data object to the list
        pyg_graphs.append(data)
    # Increment progress counter
    progress_counter += 1

    # Print progress as percentage
    progress_percentage = (progress_counter / total_groups) * 100
    print(f"Progress: {progress_percentage:.2f}% complete", end='\r')

print("Processing complete.")
