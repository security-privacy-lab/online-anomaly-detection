# First, the necessary modules are imported
import pandas
import torch
import random
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx
import networkx as nx
from networkx.readwrite import json_graph
import json

if __name__ == '__main__':
    # We first import the dataset into the code, specifically by using a pandas dataframe.
    df = pandas.read_csv("graph_creation/data/Darknet.CSV")
    # Once that's done, we convert the timestamp format to something that can be better used by pandas.
    df["Timestamp"] = pandas.to_datetime(
        df["Timestamp"], format="%d/%m/%Y %I:%M:%S %p")
    # Then create a rounded to the minute field for grouping later.
    df["timestamp_minute"] = df["Timestamp"].dt.floor("min")
    # we sort the dataframe by timestamp earliest to latest...
    df = df.sort_values(by=["Timestamp"])

    # and group the entries by minute...
    grouped = df.groupby("timestamp_minute")
    # We then grab the group length for progress tracking, but this is an optional step that can be removed.
    total_groups = len(grouped)
    progress_counter = 0
    # We then create an array for the graphs...
    pyg_graphs = []

    pg_json = {"graphs": []}

    # And now, for every minute in the dataset, we run the graph creation code.
    for timestamp_minute, group in grouped:
        # For testing, nodes will be named by their index, but you may want to use flow id instead.
        index = 0
        # Then, we create the graph.
        G = nx.DiGraph()
        # Add meta data the graph attribute.
        G.graph["timestamp"] = str(timestamp_minute)
        # And maintain the list of nodes.
        nodes_list = []
        # We establish a previous node variable to store for easier graph creation.
        previous_backward_node = None
        # Find the number of nodes.
        num_rows = list(range(2*len(group)))
        # Shuffle to randomly assign nodes to the training, validation, and testing sets.
        random.shuffle(num_rows)
        training = num_rows[:int(len(num_rows)*0.7)]
        validating = num_rows[int(len(num_rows)*0.7):int(len(num_rows)*0.85)]
        testing = num_rows[int(len(num_rows)*0.85):]

        # Now, for each entry...
        for idx, row in group.iterrows():
            # Record the source and destination IP, as well as the flow ID.
            src_ip = row["Src IP"]
            dst_ip = row["Dst IP"]
            flow_id = row[
                "Flow ID"
            ]  # Currently, this isn't being used, but again, you may want to use this for indexing?

            # Initialize node features dictionary with common attributes for both forward and backward nodes
            node_features = {
                "src_ip": src_ip,
                "dist_ip": dst_ip,
                "flow_id": flow_id
            }

            # Update the dictionary with specific attributes for the forward node
            # Another thing to note is that for pytorch-geo use, you need the features to share the same attributes, so the names are now shared, although the values
            # are being pulled from different places for forwards and backwards nodes.

            val = False
            test = False

            if index in testing:
                test = True
            elif index in validating:
                val = True

            # Create the label
            fwd_label = row["Label"].lower()
            label = 0 if (fwd_label == "nonvpn" or fwd_label ==
                          "non-tor") else 1

            forward_features = {"test": test, "id": index, "features": [
                row["Total Fwd Packet"],
                row["Fwd Packets/s"],
                row["Fwd IAT Std"],
                row["Total Length of Fwd Packet"],
                row["Fwd Packet Length Std"],
                row["Fwd Segment Size Avg"],
                row["FWD Init Win Bytes"],
                row["Fwd Packet Length Mean"],
                row["Fwd IAT Max"],
                row["Average Packet Size"],
                row["Subflow Fwd Bytes"],
                1,
            ], "val": val, "label": label, "ip": src_ip}

            node_features.update(forward_features)

            # Add the forward node to the graph
            forward_node_id = str(flow_id) + "fwd"
            G.add_node(forward_node_id, **node_features)
            nodes_list.append(forward_node_id)

            index += 1

            # Create the label
            bwd_label = row["Label"].lower()
            label = 0 if (bwd_label == "nonvpn" or bwd_label ==
                          "non-tor") else 1

            val = False
            test = False

            if index in testing:
                test = True
            elif index in validating:
                val = True

            # Update the dictionary with specific attributes for the backward node
            backward_features = {"test": test, "id": index, "features": [
                -row["Total Bwd packets"],
                -row["Bwd Packets/s"],
                -row["Bwd IAT Std"],
                -row["Total Length of Bwd Packet"],
                -row["Bwd Packet Length Std"],
                -row["Bwd Segment Size Avg"],
                -row["Bwd Init Win Bytes"],
                -row["Bwd Packet Length Mean"],
                -row["Bwd IAT Max"],
                -row["Average Packet Size"],
                -row["Subflow Bwd Bytes"],
                - 1,
            ], "val": val, "label": label, "ip": dst_ip}

            node_features.update(backward_features)

            # Add the backward node to the graph
            backward_node_id = str(flow_id) + "bwd"
            G.add_node(backward_node_id, **node_features)
            nodes_list.append(backward_node_id)

            index += 1

            # Connect the edges as appropriate (in this case, chronological order, as we are dealing with network traffic.)
            G.add_edge(forward_node_id, backward_node_id)
            # we then connect the previous backwards node to the next forward node, although we may need to change this in the future.
            if previous_backward_node is not None:
                G.add_edge(previous_backward_node, forward_node_id)

            previous_backward_node = backward_node_id

            # Convert the NetworkX graph to a PyTorch Geometric Data object
            data = from_networkx(G)

            # Append the PyG Data object to the list
            pyg_graphs.append(data)

        pg_json["graphs"].append(json_graph.node_link_data(G))
        # All of the following code is optional and only exists to be able to track the progress of the program.
        # Increment progress counter
        progress_counter += 1

        # Print progress as percentage
        progress_percentage = (progress_counter / total_groups) * 100
        print(f"Progress: {progress_percentage:.2f}% complete", end="\r")

    print("\033[K", end="\r")
    print("Processing complete.")
    print("Converting into JSON file...")

    with open("graph_creation/processed_data/pygraphs.json", 'w+') as f:
        json.dump(pg_json, f)

    print("Convertion complete")
