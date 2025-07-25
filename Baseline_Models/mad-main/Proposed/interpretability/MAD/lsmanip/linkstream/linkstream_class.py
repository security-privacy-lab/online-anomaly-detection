class LinkStream:
    """ Class to represent a link stream
        
    :param isweighted: 'True' if triplets have associated weights, 'False' otherwise
    :param size: number of triplets in the link stream
    :param num_nodes: number of nodes in the link stream
    :param max_time: maximum time contained in a triplet
    :param min_time: minimum time contained in a triplet
    :param num_times: number of different time stamps
    """
    def __init__(self):
        # Data : a set of triplets
        self._data = dict()

        # Properties 
        self.isweighted = False

        # Basic stats
        self.size = 0 # Number of triplets
        self.num_nodes = None # Number of nodes
        self.max_time = None # Maximum time tick
        self.min_time = None # Minimum time tick
        self.num_times = None # Number of different time ticks

    # Construct link stream from a weight function
    @classmethod
    def from_weightfn(cls, weightfn):
        link_stream = cls()
        for triplet in weightfn:
            link_stream.add_triplet( triplet )
            link_stream.set_triplet_weight(triplet, weightfn[triplet])
        link_stream.set_stats()
        return link_stream

    # Construct link stream from a set of triplets
    @classmethod
    def from_triplets(cls, triplets):
        link_stream = cls()
        for triplet in triplets:
            link_stream.add_triplet( triplet )
        link_stream.set_stats()
        return link_stream           
    
    # Add a new triplet into the link stream
    def add_triplet(self, triplet):
        """ Add a triplet (u,v,t) """
        self._data[ triplet ] = 1
    
    # Set the weight of a given triplet
    def set_triplet_weight(self, triplet, weight):
        """ Set a triplet weight """
        self._data[ triplet ] = weight
        self.isweighted = True
    
    # Return a set of triplets
    def to_triplets(self):
        """ Link stream represented as a set of triplets
        """
        return set(self._data.keys())
    
    # Return a time series of graphs
    def to_tsgraphs(self):
        """ Link stream represented as a time series of graphs
        """
        tgraph = dict()
        for u,v,t in self._data.keys(): 
            if t not in tgraph: tgraph[t] = dict()
            tgraph[t][(u,v)] = self._data[(u,v,t)]
        return tgraph
        
    # Return indicator times of edges 
    def to_edgets(self):
        """ Link stream represented as time series of relations
        """
        edgets = dict()
        for u,v,t in self._data.keys():
            if (u,v) not in edgets: edgets[(u,v)] = dict()
            edgets[(u,v)][t] = self._data[(u,v,t)]
        return edgets
    
    # Return the weight function (t,u,v) -> w
    def to_weightfn(self):
        """ Link stream represented as a function (t,u,v) -> w
        """
        return self._data
    
    # Set link stream statistics
    def set_stats(self):
        """ Compute basic statistics about a link stream
        """
        # Store nodes and times
        nodes = set()
        times = set()

        # Count triplets and times
        triplet_count = 0
        max_time = None
        min_time = None

        for u,v,t in self._data.keys():
            # Update nodes and times
            nodes.add( u )
            nodes.add( v )
            times.add( t )
            # Set times
            if max_time == None: max_time = t
            if min_time == None: min_time = t   
            if t < min_time : min_time = t
            if t > max_time : max_time = t
            # Triplet counter
            triplet_count += 1  

        self.size = triplet_count # Set size
        self.num_nodes = len(nodes) # Set num_nodes
        self.num_times = len(times) # Set num_times
        self.max_time = max_time # Set max_time
        self.min_time = min_time # Set min_time
