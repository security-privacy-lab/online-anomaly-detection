import json
import argparse
import networkx as nx
import random
from networkx.readwrite import json_graph
from pathlib import Path

MODELS = ["gin", "graphsage"]


def process_data(models):
    funcs_to_process = MODELS if "all" in models else models
    for model in funcs_to_process:
        func = globals()[model]
        func()


def maxgraph():
    graph_lst = json.load(
        open("graph_creation/processed_data/pygraphs.json"))
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

    # Initialize the dict for id_map and class_map needed for graphSAGE
    id_map = {}
    class_map = {}

    index = 0


def graphsage():
    # Loop through all nodes and populate the id_map and class_map
    for node in max_graph["nodes"]:
        id_map[node["id"]] = index
        class_map[node["id"]] = node["label"]
        index += 1

    # Dump id_map and class_map into JSON files
    with open("graph_creation/processed_data/max_pygraphs-id_map.json", "w+") as f:
        json.dump(id_map, f)
    with open("graph_creation/processed_data/max_pygraphs-class_map.json", "w+") as f:
        json.dump(class_map, f)


def gin():
    print("Processing data for GIN...")

    path = Path(__file__).parent / "processed_data" / "pygraphs.json"

    with open(path, "r") as f:
        data = json.load(f)

    nx_graphs = []
    for graph in data["graphs"]:
        nx_graphs.append(json_graph.node_link_graph(graph))

    print(len(nx_graphs))
    random.shuffle(nx_graphs)
    for graph in nx_graphs:
        for node in graph.nodes:
            in_edges = graph.in_edges(node)
            out_edges = graph.out_edges(node)

    print("Data for GIN successfully processed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Process the general NX Graph object (pygraphs.json) into various formats suitable to be ran by the various baseline models.')

    parser.add_argument('--models', choices=MODELS+["all", "maxgraph"], default='all',
                        nargs='+', help="the model you need to generate data for [all, gin, graphsage, maxgraph].")
    args = parser.parse_args()
    models = args.models
    process_data(models)
