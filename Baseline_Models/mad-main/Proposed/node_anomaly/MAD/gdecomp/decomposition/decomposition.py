import pywt
import math

# Decomposition coefficients of a graph
def decomposition_coefficients(graph, dictionary):
	
	""" Decomposition coefficients of a graph under a given dictionary.

		The method assumes that the provided graph is decomposable under the 
		provided dictionary.

		:param: graph: Input graph
		:param: dictionary: Input dictionary

	"""
	
	# Get the leaf nodes of the tree into a vector suited for wavelet transform
	data = tree_leaves(graph, dictionary)

	# Haar wavelet transforms computes the coefficients
	coeffs = pywt.wavedec(data, 'haar')
	return coeffs

# Vector representing the mapping of graph to tree leave nodes
def tree_leaves(graph, dictionary):
	
	# Data in approapiate format  G : edge -> weight
	graphfn = graph.to_weightfn()

	# allocate memory: a power of two where dictionary elements fit
	if dictionary.get_size() == 0:
		memory_size = 0
	else: 
		memory_size = 2**math.ceil(math.log2(dictionary.get_size()))
	data = [0]*memory_size

	# Set the leaf nodes of the decomposition tree in the right order 
	for edge in graphfn:
		data[ dictionary.get_index(edge) ] = graphfn[edge]

	return data

# Scaling coefficients at all levels
def multiscale_approximation(dec):
	scaling = [dec[0]]
	for l in range(1, len(dec)):
		scaling.append( pywt.idwt(scaling[-1], dec[l], 'haar') )
	
	return scaling
