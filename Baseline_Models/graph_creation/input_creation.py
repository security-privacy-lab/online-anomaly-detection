import json
import argparse
import networkx as nx
import numpy as np
import os
from networkx.readwrite import json_graph
from pathlib import Path

MODELS = ["gin", "graphsage", "gat"]
K = 10


def process_data(models):
    funcs_to_process = MODELS if "all" in models else models
    for model in funcs_to_process:
        func = globals()[model]
        func()


def maxgraph():
    graph_lst = json.load(open("graph_creation/processed_data/pygraphs.json"))
    # Initialize counter and dict to keep track of the largest graph
    num_nodes = 0
    max_graph = {}

    # Now iterate through all nodes to find the largest graph
    for graph in graph_lst["graphs"]:
        if len(graph["nodes"]) > num_nodes:
            num_nodes = len(graph["nodes"])
            max_graph = graph

    # Dump the graph into a JSON file
    with open("graph_creation/processed_data/max_pygraphs-G.json", "w+") as f:
        json.dump(max_graph, f)


def maxgraphs():
    graph_lst = json.load(open("graph_creation/processed_data/pygraphs.json"))

    sorted_graphs = sorted(
        graph_lst["graphs"], key=lambda x: len(x["nodes"]), reverse=True
    )
    print("There are " + str(len(sorted_graphs)) + " graphs in the dataset")

    top_graphs = sorted_graphs[:625]

    merged_graph = {
        "nodes": [],
        "links": [],
    }
    for graph in top_graphs:
        merged_graph["nodes"].extend(graph["nodes"])
        if "links" in graph:
            merged_graph["links"].extend(graph["links"])

    merged_graph["nodes"] = list(
        {node["id"]: node for node in merged_graph["nodes"]}.values()
    )

    node_ids = {node["id"] for node in merged_graph["nodes"]}
    merged_graph["links"] = list(
        {
            (link["source"], link["target"]): link
            for link in merged_graph["links"]
            if link["source"] in node_ids and link["target"] in node_ids
        }.values()
    )

    with open("graph_creation/processed_data/max_pygraphs-G.json", "w+") as f:
        json.dump(merged_graph, f)


def allgraphs():
    graph_lst = json.load(open("graph_creation/processed_data/pygraphs.json"))
    graphs = sorted(graph_lst["graphs"], key=lambda x: len(x["nodes"]), reverse=True)
    merged_graph = {
        "nodes": [],
        "links": [],
    }

    for graph in graphs:
        merged_graph["nodes"].extend(graph["nodes"])
        if "links" in graph:
            merged_graph["links"].extend(graph["links"])

    merged_graph["nodes"] = list(
        {node["id"]: node for node in merged_graph["nodes"]}.values()
    )

    node_ids = {node["id"] for node in merged_graph["nodes"]}
    merged_graph["links"] = list(
        {
            (link["source"], link["target"]): link
            for link in merged_graph["links"]
            if link["source"] in node_ids and link["target"] in node_ids
        }.values()
    )
    with open("graph_creation/processed_data/max_pygraphs-G.json", "w+") as f:
        json.dump(merged_graph, f)


def graphsage():
    allgraphs()
    # maxgraphs()
    # maxgraph()
    graph = json.load(open("graph_creation/processed_data/max_pygraphs-G.json", "r"))
    id_map = {}
    class_map = {}

    index = 0
    for node in graph["nodes"]:
        id_map[node["id"]] = index
        class_map[node["id"]] = node["label"]
        index += 1

    with open("graph_creation/processed_data/max_pygraphs-id_map.json", "w+") as f:
        json.dump(id_map, f)
    with open("graph_creation/processed_data/max_pygraphs-class_map.json", "w+") as f:
        json.dump(class_map, f)


def gin():
    print("Processing data for GIN...")

    # Locating the pygraphs.json file and loading it into an NX Graph object
    path = Path(__file__).parent / "processed_data" / "pygraphs.json"

    with open(path, "r") as f:
        data = json.load(f)

    nx_graphs = []

    init_node_features = nx.get_node_attributes(
        json_graph.node_link_graph(data["graphs"][0]), "features"
    )[0]

    max_vector = np.abs(np.array(init_node_features))
    min_vector = np.abs(np.array(init_node_features))

    for graph in data["graphs"]:
        nx_graph = json_graph.node_link_graph(graph)
        nx_graphs.append(nx_graph)

        features = nx.get_node_attributes(nx_graph, "features")

        for node in nx_graph.nodes:
            max_vector = np.maximum(max_vector, np.abs(np.array(features[node])))
            min_vector = np.minimum(min_vector, np.abs(np.array(features[node])))

    max_vector = max_vector[:-1]
    min_vector = min_vector[:-1]

    num_graphs = len(nx_graphs)

    # Looping through all the graphs and assigning them to folds and
    # coverting them into appropriate format

    file = f"{num_graphs}\n"

    for graph in nx_graphs:

        features = nx.get_node_attributes(graph, "features")

        num_nodes = len(graph.nodes)

        labels = nx.get_node_attributes(graph, "label")

        graph_label = 1 if any(node_label == 1 for node_label in labels.values()) else 0

        # Adding the graph metadata
        file += f"{num_nodes} {graph_label}\n"

        # Now adding the information of each node
        for node in graph.nodes:
            neighbors = list(nx.all_neighbors(graph, node))
            node_features = np.array(features[node])[:-1]
            direction = -1 if np.any(node_features < 0) else 1
            node_features = (
                direction
                * (np.abs(node_features) - min_vector)
                / (max_vector - min_vector)
            )
            file += f"{node} {len(neighbors)} {' '.join([str(n) for n in neighbors])} {' '.join([str(f) for f in node_features])}\n"

    directory = Path(__file__).parent / "processed_data" / "gin_data"

    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(directory / "data.txt", "w+") as f:
        f.write(file)

    gin_directory = Path(__file__).parent.parent / "GIN" / "dataset" / "DARKNET"
    if not os.path.exists(gin_directory):
        os.makedirs(gin_directory)

    with open(gin_directory / "DARKNET.txt", "w+") as f:
        f.write(file)

    print("Data for GIN successfully processed")

def gat():
    print("Processing data for GAT...")
    graph_lst = json.load(open("graph_creation/processed_data/pygraphs.json", "r"))
    graphs = sorted(graph_lst["graphs"], key=lambda x: len(x["nodes"]), reverse=True)
    merged_graph = {
        "nodes": [],
        "links": [],
    }

    for graph in graphs:
        merged_graph["nodes"].extend(graph["nodes"])
        if "links" in graph:
            merged_graph["links"].extend(graph["links"])
    
    merged_graph["nodes"] = list(
        {node["id"]: node for node in merged_graph["nodes"]}.values()
    )

    node_ids = {node["id"] for node in merged_graph["nodes"]}
    merged_graph["links"] = list(
        {
            (link["source"], link["target"]): link
            for link in merged_graph["links"]
            if link["source"] in node_ids and link["target"] in node_ids
        }.values()
    )

    id_map = {}
    class_map = {}

    index = 0
    for node in merged_graph["nodes"]:
        id_map[node["id"]] = index
        class_map[node["id"]] = node["label"]
        index += 1
    
    with open("graph_creation/processed_data/gat_pygraphs.json", "w+") as f:
        json.dump(merged_graph, f)
    with open("graph_creation/processed_data/gat_id_map.json", "w+") as f:
        json.dump(id_map, f)
    with open("graph_creation/processed_data/gat_class_map.json", "w+") as f:
        json.dump(class_map, f)

    print("Data for GAT successfully processed")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process the general NX Graph object (pygraphs.json) into various formats suitable to be ran by the various baseline models."
    )

    parser.add_argument(
        "--models",
        choices=MODELS + ["all", "maxgraph"],
        default="all",
        nargs="+",
        help="the model you need to generate data for [all, gin, graphsage, maxgraph].",
    )
    args = parser.parse_args()
    models = args.models
    process_data(models)
