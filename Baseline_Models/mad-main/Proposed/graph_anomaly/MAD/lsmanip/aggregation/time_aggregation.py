# Get the aggregated graph
def aggregated_graph(linkstream):
	""" Aggregate link stream into a single graph

	:param linkstream: a linkstream object
	:type linkstream: :class: 'LinkStram'
	:return: A function edge -> weight
	:rtype: dict[Tuple[str,str], float]
	"""
	graph = dict()
	weightfn = linkstream.to_weightfn()
	for triplet, value in weightfn.items():
		u,v,_ = triplet
		if (u,v) not in graph: graph[(u,v)] = 0
		graph[(u,v)] += value
	return graph
