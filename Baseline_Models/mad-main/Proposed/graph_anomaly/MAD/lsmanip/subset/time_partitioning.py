from .. import LinkStream

# Sublink-stream that contains a set of nodes
def time_partition(linkstream, interval):
	""" Link stream restricted to a specified time interval [t_min, tmax)
		
	:param linkstream: A link stream object
	:param interval: interval of times to retain
	:type linkstream: :class: LinkStream
	:type interval: list[t_min, t_max]
	:return: A link stream object 
	:rtype: :class: LinkStream
	"""
	weightfn = linkstream.to_weightfn() # Get the link stream  weight funciton
	sub_linkstream = LinkStream() # Object to represent partitioned linkstream
	for triplet, value in weightfn.items():
		_,_,t = triplet
		if interval[0] <= t < interval[1]:  # condition
			sub_linkstream.add_triplet( triplet )
			if linkstream.isweighted: sub_linkstream.set_triplet_weight( triplet, value )
	
	# Set members of sub linkstream
	sub_linkstream.isweighted = linkstream.isweighted
	sub_linkstream.set_stats()
	return sub_linkstream
