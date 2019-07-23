import numpy as np
import numpy.linalg as LA
from scipy.linalg import fractional_matrix_power

class Graph():
	'''
	Class for unweighted graph. The graph is not necessarily undirected,
	but by default is set to be undirected. The graph is also simple graph
	(no self-loop and at most one edge connecting each pair of vertex).

	Parameters
	----------

	V : int
	   The number of vertices.
	
	E : int, optional, default 0
	   The number of edges.
	
	vertex_label : set, optional, default None
	   The label of each vertex. Have to be specified if the label is not int,
	   in order to work properly. If not specified (None), the vertex label
	   is set to Python set {0, 1, 2, ..., V - 1}. 
	
	edge_list : list of tuples of int or string, optional, default empty list
	   List containing tuples [(u_0, v_0), (u_1, v_1), ... , (u_(E-1), v_(E-1))]
	   where u_i and v_i is label of any vertex. If (u, v) present in edge_list,
	   it means that vertex with label u is adjacent with vertex with label v.
	   If the graph is undirected, it also implies that vertex with label v is
	   adjacent with vertex with label u. Else, a two-way arc is considered as 
	   an edge.
	
	undirected : boolean, optional, default True
	   True if the graph is undirected, False if directed.

	Examples
	--------
	
	>>> V, E = 3, 2
	>>> edge_list = [(0, 1), (1, 2)]
	>>> G = Graph(V, E, edge_list=edge_list)
	>>> print(G.vertex_label())
	{0, 1, 2}

	>>> edge_list = [('a', 'b'), ('c', 'd'), ('c', 'b')]
	>>> vertex_label = {'a', 'b', 'c', 'd'}
	>>> G = Graph(4, 3, vertex_label, edge_list)
	>>> print(G.vertex_label())
	{'d', 'b', 'a', 'c'}

	'''
	def __init__(self, V, E=0, vertex_label=None, edge_list=[], undirected=True):
		if len(edge_list) != E:
			raise ValueError("The length of edge list must be equal to E.")
		
		self.__V = V
		self.__E = E
		self.__edge_list = edge_list
		self.__undirected = undirected

		if vertex_label == None:
			self.__vertex_label = set(range(V))
		else:
			if len(vertex_label) != V:
				raise ValueError("The length of vertex_label must be equal to V.")
			else:
				self.__vertex_label = vertex_label

	def get_V(self):
		'''
		Return the number of vertex of the graph.
		'''
		return self.__V

	def get_E(self):
		'''
		Return the number of edge of the graph.
		'''
		return self.__E

	def is_undirected(self):
		'''
		Return True if the graph is undirected, False if directed.
		'''
		return self.__undirected

	def vertex_label(self):
		'''
		Return Python set containing the label of each vertex.
		'''
		return self.__vertex_label

	def edge_list(self, numpify=False):
		'''
		Return the edge list of the graph:
		[(u_0, v_0), (u_1, v_1), ... , (u_(E-1), v_(E-1))].


		Parameters
		----------

		numpify : boolean, optional, default False
		   If True, return NumPy array of shape (2, E):
		   [[u_0, u_1, u_2, ... , u_(E-1)],
		    [v_0, v_1, v_2, ... , v_(E-1)]]
		   else return edge list of the graph in form of list of tuples.
		
		'''
		if numpify:
			return np.array(self.__edge_list).T
		return self.__edge_list

	def adj_matrix(self):
		'''
		Return the adjacency matrix of the graph. Adjacency matrix A of the graph
		is represented as NumPy array with shape of (V, V). The indexing of
		NumPy array follows a mapping from label_map. See examples for details.

		Returns
		-------
		
		Tuple of (A, label_map)

		   A : NumPy array of shape (V, V), the adjacency matrix of the graph

		   label_map : dict which maps vertex label to index of A

		Examples
		--------

		>>> edge_list = [('a', 'b'), ('c', 'd'), ('c', 'b')]
		>>> vertex_label = {'a', 'b', 'c', 'd'}
		>>> G = Graph(4, 3, vertex_label, edge_list)
		>>> G.adj_matrix()
		(array([[0, 0, 1, 1],
		       [0, 0, 0, 1],
		       [1, 0, 0, 0],
		       [1, 1, 0, 0]]), {'c': 0, 'a': 1, 'd': 2, 'b': 3})

		Indices of A correspond to label 'c', 'a', 'd', and 'b' respectively.
		The value of A[0, 2] is 1 because the vertex 'c' (correspond to index 0)
		is adjacent to vertex 'd' (correspond to index 2).
		
		'''
		A = np.zeros((self.__V, self.__V), dtype=np.int)
		label_map = {list(self.__vertex_label)[i]: i for i in range(self.__V)}

		if self.__E == 0:
			return (A, label_map)

		edge_list = [(label_map[u], label_map[v]) for (u, v) in self.__edge_list]
		edge_list = np.array(edge_list).T

		A[tuple(edge_list)] = 1
		if self.__undirected:
			A[tuple(edge_list[::-1])] = 1

		return (A, label_map)

	def adj_list(self):
		'''
		Return the adjacency list of the graph. Adjacency list of the graph is
		represented by Python dict with vertex label as the keys. The adjacency
		list maps vertex label v to Python set of neighboring vertices of v.

		Returns
		-------

		adj_list : dict, adjacency list of the graph

		Examples
		--------

		>>> V, E = 4, 5
		>>> edge_list = [(0, 1), (1, 2), (2, 3), (0, 2), (1, 3)]
		>>> G = Graph(V, E, edge_list=edge_list)
		>>> G.adj_list()
		{0: {1, 2}, 1: {0, 2, 3}, 2: {0, 1, 3}, 3: {1, 2}}

		>>> edge_list = [('a', 'b'), ('c', 'd'), ('c', 'b')]
		>>> vertex_label = {'a', 'b', 'c', 'd'}
		>>> G = Graph(4, 3, vertex_label, edge_list)
		>>> G.adj_list()
		{'d': {'c'}, 'b': {'c', 'a'}, 'c': {'d', 'b'}, 'a': {'b'}}

		'''
		adj_list = {v : set() for v in self.__vertex_label}

		for e1, e2 in self.__edge_list:
			adj_list[e1].add(e2)
			if self.__undirected:
				adj_list[e2].add(e1)
		
		return adj_list

	def depth_first_search(self, v):
		'''
		Perform iterative Depth-First Search (DFS) from vertex
		with label v in the graph. Return a list containing label of
		vertices traversed during DFS, in order.

		Parameters
		----------

		v : label of starting vertex

		Returns
		-------

		traversed : list containing label of vertices traversed
		   during DFS, in order
		'''
		traversed = [v]
		stack = [v]
		adj_list = self.adj_list()

		while len(stack):
			v = stack[-1]
			if v not in traversed:
				traversed.append(v)

			removed = True
			for next in adj_list[v]:
				if next not in traversed:
					stack.append(next)
					removed = False
					break
			
			if removed:
				stack.pop()

		return traversed

	def connected_components(self):
		'''
		Return connected components of the graph. The connected components
		are represented in list of Graph objects. The list is sorted by the
		number of vertices of its elements in ascending order. If tied,
		it sorted by the number of the edges.

		Returns
		-------

		connected_components : list of Graph objects, the connected components
		   of the graph
		'''
		vertex_labels = []

		all_node = self.__vertex_label
		visited = set() # set of visited vertex

		while len(visited) < self.__V:
			# select an arbitrary unvisited vertex
			v = all_node.difference(visited).pop()
			visited.add(v)

			traversed = self.depth_first_search(v)
			visited.update(traversed)
			vertex_labels.append(traversed)

		# build the connected components: list of Graph objects
		adj_list = self.adj_list()
		connected_components = []

		for vertex_set in vertex_labels:
			edge_list = []
			for u in vertex_set:
				for v in adj_list[u]:
					if (v, u) not in edge_list:
						edge_list.append((u, v))

			connected_components.append(\
				Graph(len(vertex_set), len(edge_list), vertex_set, edge_list)\
			)

		# sort the connected components
		def key(G):
			return (G.get_V(), G.get_E())

		connected_components = sorted(connected_components, key=key)
		return connected_components

	def spectral_feature(self, k=None):
		'''
		Extract spectral feature of the graph. See [1] for details.

		Parameters
		----------

		k : int, desired length of the feature vector

		Returns
		-------
		
		sf : spectral feature, numpy array of shape (k, )

		----------

		References:
		[1] N. de Lara and E. Pineau, "A Simple Baseline Algorithm
		for Graph Classification" In Relational Representation Learning,
		NIPS 2018 Workshop, Montréal, Canada.
		'''

		# get the largest connected component
		G = self.connected_components()[-1]

		A, mapper = G.adj_matrix()
		
		# create matrix of node degrees
		v1 = np.ones((G.get_V(), ), dtype=np.int)
		D = np.diag(A @ v1)

		# create normalized Laplacian matrix of the graph
		I = np.eye(G.get_V(), dtype=np.int)
		P = fractional_matrix_power(D, -0.5)
		L = I - (P @ A @ P)

		# compute the eigenvalue of the Laplacian matrix
		# and create the feature vector
		eigvals = LA.eigvals(L)
		eigvals.sort()

		if k == None:
			return eigvals

		sf = eigvals[:k]

		# pad the feature vector if the graph has less than k vertices
		if G.get_V() < k:
			pad = np.zeros((k - len(sf) + 1, ))
			sf = np.append(sf[:-1], pad)

		return sf

def random_graph(V, E, random_state=None):
	'''
	Return a random Graph object with V number of vertices and E number
	of edges.

	Parameters
	----------

	V : int, the number of vertex

	E : int, the number of edge

	random_state : int, default None, for reproducibility purpose

	Returns
	-------

	Graph object with V number of vertices and E number of edges.

	'''
	edge_list = [(i, j) for i in range(V) for j in range(i + 1, V)]
	
	np.random.seed(random_state)
	np.random.shuffle(edge_list)
	
	return Graph(V, E, edge_list=edge_list[:E])
