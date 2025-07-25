from .. import LinkStream

# Sublink-stream that contains a set of nodes
def node_partition(linkstream, nodes):
	""" Link stream restricted to a specified set of nodes
		
	:param linkstream: A link stream object
	:param nodes: Nodes to be retained
	:type linkstream: :class: LinkStream
	:type nodes: set[str]
	:return: A link stream object 
	:rtype: :class: LinkStream
	"""
	weightfn = linkstream.to_weightfn() # Get the link stream  weight funciton
	sub_linkstream = LinkStream() # Object to represent partitioned linkstream
	for triplet, value in weightfn.items():
		u,v,_ = triplet
		if u in nodes and v in nodes:  # condition
			sub_linkstream.add_triplet( triplet )
			if linkstream.isweighted: sub_linkstream.set_triplet_weight( triplet, value )
	
	# Set members of sub linkstream
	sub_linkstream.isweighted = linkstream.isweighted
	sub_linkstream.set_stats()
	return sub_linkstream

# Sublink-stream restricted to a bi-partite sub-graph 
def bipartite_partition(linkstream, origin_set, destin_set):
	""" Link stream restricted to sets of origin and destination nodes
		
	:param linkstream: A link stream object
	:param origin_set: set of origin ndoes to retain 
	:param destin_set: set of destination ndoes to retain 
	:type linkstream: :class: LinkStream
	:type origin_set: set[str]
	:type destin_set: set[str]
	:return: A link stream object 
	:rtype: :class: LinkStream
	"""
	weightfn = linkstream.to_weightfn() # Get the link stream  weight funciton
	sub_linkstream = LinkStream() # Object to represent partitioned linkstream
	for triplet, value in weightfn.items():
		u,v,_ = triplet
		if u in origin_set and v in destin_set:  # condition
			sub_linkstream.add_triplet( triplet )
			if linkstream.isweighted: sub_linkstream.set_triplet_weight( triplet, value )
	
	# Set members of sub linkstream
	sub_linkstream.isweighted = linkstream.isweighted
	sub_linkstream.set_stats()
	return sub_linkstream
