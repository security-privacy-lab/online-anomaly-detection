import math 
import random

##########################
### MAPPER BASE CLASS ####
##########################
class Mapper:
	def _np2(self, x): return 1 if x == 0 else 2**math.ceil(math.log2(x))
	def get_size(self): return 0
	def index(self, query_edge): return 0

########################################
### MAPPER : DICTIONARY OF BICLIQUES ### 

###### ORDER BY RANK ########
class MapperBiclique_byRank(Mapper):
	def __init__(self, origin, destin, rank):
		self._origin = origin
		self._destin = destin
		self._size_origin = len(origin)
		self._size_destin = len(destin)

		sorted_edges = sorted(rank, key=rank.get, reverse=True)
		self._mapping = {e:idx for idx,e in enumerate(sorted_edges)}
		for u in self._origin:
			for v in self._destin:
				if (u,v) not in self._mapping: 
					self._mapping[(u,v)] = len(self._mapping)

	def index(self, query_edge):
		return self._mapping[query_edge]

	def get_size(self):
		return len(self._mapping)

###### ORDER BY NODES ########
class MapperBiclique_byNode(Mapper):
	def __init__(self, origin, destin, pad):
		self._origin_order = dict()
		self._destin_order = dict() 
		self._size_origin = len(origin)
		self._size_destin = len(destin)
		self._pad = pad
	
	def index(self, query_edge):
		u,v = query_edge
		if self._pad:
			new_size = self._np2(self._size_destin)
			return self._origin_order[u]*new_size + self._destin_order[v]
		else:
			return self._origin_order[u]*self._size_destin + self._destin_order[v] 

	def get_size(self): return self._size_origin * self._size_destin

# DERIVED CLASSES #
class MapperBiclique_byNodeRnd(MapperBiclique_byNode):	 # Child class
	def __init__(self, origin, destin, pad=False, seed=None):
		super().__init__(origin, destin, pad)	
		
		# Set up the order of nodes (RANDOM)
		if seed != None: random.seed(seed)
		rnd_origin = list(origin)
		rnd_destin = list(destin)
		random.shuffle(rnd_origin)
		random.shuffle(rnd_destin)
		self._origin_order = {u:idx for idx,u in enumerate(rnd_origin)}
		self._destin_order = {u:idx for idx,u in enumerate(rnd_destin)}

class MapperBiclique_byNodeRank(MapperBiclique_byNode):	 # Child class
	def __init__(self, origin_rank, destin_rank, pad=False):
		super().__init__(origin_rank, destin_rank, pad)	
		
		# Set up the order of nodes (By Rank)
		self._origin_order = origin_rank
		self._destin_order = destin_rank

########################################
### MAPPER : DICTIONARY OF SUBGRAPHS ### 
class MapperSubgraph_byEdgeRnd(Mapper):
	def __init__(self, edgespace, seed=None):
		
		# Set the order of edges at random
		if seed != None: random.seed(seed)
		rnd_edgespace = list(edgespace)
		random.shuffle(rnd_edgespace)
		self._mapping = {e:idx for idx,e in enumerate(rnd_edgespace)}

	def index(self, query_edge):
		return self._mapping[query_edge]

	def get_size(self):
		return len(self._mapping)

class MapperSubgraph_byRank(Mapper):
	def __init__(self, edgespace, rank):
		sorted_edges = sorted(rank, key=rank.get, reverse=True)
		self._mapping = {e:idx for idx,e in enumerate(sorted_edges)}

		# Add edgespace not ranked
		for e in edgespace:
			if e not in self._mapping: self._mapping[e] = len(self._mapping)

	def index(self, query_edge):
		return self._mapping[query_edge]

	def get_size(self):
		return len(self._mapping)

##############################
### DICTIONARY BASE CLASS  ### 
##############################
class Dictionary:	
	def __init__(self):
		self._origin = set()
		self._destin = set()
		self._edgelist = set()
		self.Mapper = Mapper()
	
	def get_index(self, query_edge):
		return self.Mapper.index(query_edge)
	
	def get_size(self):
		return self.Mapper.get_size()

###############################
### DICTIONARY OF SUBGRAPHS ### 
class DictionarySubgraph(Dictionary):
	def __init__(self, edgespace):
		Dictionary.__init__(self)
		self._edgelist = edgespace
		self.order_default()
	
	def order_default(self):
		self.order_edges_rnd()
		
	def order_edges_rnd(self, seed=None):
		self.Mapper = MapperSubgraph_byEdgeRnd(self._edgelist, seed)

	def order_rank(self, rank):
		self.Mapper = MapperSubgraph_byRank(self._edgelist, rank)

	
###############################
### DICTIONARY OF BICLIQUES ### 
class DictionaryBiclique(Dictionary):
	def __init__(self, origin, destin):
		Dictionary.__init__(self)
		self._origin = origin
		self._destin = destin
		self.order_default()
	
	def order_default(self):
		self.order_nodes_rnd()
		
	def order_nodes_rnd(self, pad=False, seed=None): 
		self.Mapper = MapperBiclique_byNodeRnd(self._origin, self._destin, pad, seed)
	
	def order_nodes_rank(self, rank_origin, rank_destin, pad=False): 
		self.Mapper = MapperBiclique_byNodeRank(rank_origin, rank_destin, pad)
	
	def order_rank(self, rank):
		self.Mapper = MapperBiclique_byRank(self._origin, self._destin, rank)
