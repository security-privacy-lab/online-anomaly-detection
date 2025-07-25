from .graph_class import Graph

# Graph restricted to a clique of nodes
def node_partition(graph, nodes):
	weightfn = graph.to_weightfn()	
	subgraph = Graph()
	for e in weightfn:
		if e[0] in nodes and e[1] in nodes:
			subgraph.add_edge(e)
			subgraph.set_edge_weight(e, weightfn[e])
	return subgraph

# Graph restricted to a bipartite subgraph
def bipartite_partition(graph, origin_set, destin_set):
	weightfn = graph.to_weightfn()
	subgraph = Graph()
	for e in weightfn:
		if e[0] in origin_set and e[1] in destin_set:
			subgraph.add_edge(e)
			subgraph.set_edge_weight(e, weightfn[e])
	return subgraph

# Graph restricted to the edges of a given subgraph
def subgraph_partition(graph, subgraph):
	weightfn = graph.to_weightfn()
	restriction_set = subgraph.to_edgelist()
	restricted_graph = Graph()
	for e in weightfn():
		if e in restriction_set:
			restricted_graph.add_edge(e)
			restricted_graph.set_edge_weight(e, weightfn[e])
	return restricted_graph
