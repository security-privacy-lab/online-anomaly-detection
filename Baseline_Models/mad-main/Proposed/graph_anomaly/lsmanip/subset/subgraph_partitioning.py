from .. import LinkStream

# Sublink-stream that contains a set of nodes
def subgraph_partition(linkstream, subgraph):
	""" Link stream restricted to a specified subgraph
		
	:param linkstream: A link stream object
	:param subgraph: set of edges to be retained
	:type linkstream: :class: LinkStream
	:type subgraph: Set[Tuple[str,str]]
	:return: A link stream object 
	:rtype: :class: LinkStream
	"""
	weightfn = linkstream.to_weightfn() # Get the link stream  weight funciton
	sub_linkstream = LinkStream() # Object to represent partitioned linkstream
	for triplet, value in weightfn.items():
		u,v,_ = triplet
		if (u,v) in subgraph:  # condition
			sub_linkstream.add_triplet( triplet )
			if linkstream.isweighted: sub_linkstream.set_triplet_weight( triplet, value )
	
	# Set members of sub linkstream
	sub_linkstream.isweighted = linkstream.isweighted
	sub_linkstream.set_stats()
	return sub_linkstream
