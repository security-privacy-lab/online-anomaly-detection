import json

if __name__ == "__main__":
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
    with open("max_pygraphs.json", "w+") as f:
        json.dump(max_graph, f)

    # Initialize the dict for id_map and class_map needed for graphSAGE
    id_map = {}
    class_map = {}

    index = 0

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
