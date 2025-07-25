class Graph:
	
	def __init__(self):	
		# private members
		self._data = dict()
		self._origin_set = set()
		self._destin_set = set()
	
	@classmethod
	def from_networkx(cls, nxgraph):
		graph = cls()
		for edge in nxgraph.edges():
			graph.add_edge(edge)
			edata = nxgraph.get_edge_data(edge[0],edge[1])
			if 'weight' in edata: 
				graph.set_edge_weight(edge, edata['weight'])
		return graph
	
	@classmethod
	def from_edgelist(cls, edgelist):
		graph = cls()
		for edge in edgelist: graph.add_edge(edge)
		return graph

	@classmethod
	def from_weightfn(cls, weightfn):
		graph = cls()
		for edge in weightfn:
			graph.add_edge(edge)
			graph.set_edge_weight(edge, weightfn[edge])
		return graph
	
	def add_edge(self, edge):
		self._data[ edge ] = 1
		self.add_origin_node(edge[0])
		self.add_destin_node(edge[1])
	
	def set_edge_weight(self, edge, weight):
		self._data[ edge ] = weight
	
	def add_node(self, node):
		self.add_origin_node(node)
		self.add_destin_node(node)
	
	def add_origin_node(self, node):
		self._origin_set.add(node)
		
	def add_destin_node(self, node):
		self._destin_set.add(node)
			
	def to_edgelist(self):
		return set(self._data.keys())
	
	def to_weightfn(self):
		return self._data
	
	def get_nodes(self):
		return self._origin_set.union(self._destin_set)
	
	def get_weight(self, edge):
		return self._data[edge]
	
	def get_origin_nodes(self):
		return self._origin_set

	def get_destin_nodes(self):
		return self._destin_set
	
	def get_size(self):
		return len(self._data)
