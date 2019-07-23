from Graph import random_graph

# create a random graph with 6 nodes and 4 edges representated by its edge list
V, E, random_state = 6, 4, 1
G = random_graph(V, E, random_state=random_state)
print('Original graph:')
print('Vertex label:', G.vertex_label())
print('Edge list:', G.edge_list())

# look at the largest connected component of the graph,
H = G.connected_components()[-1]
print('\nLargest conn. component:')
print('Vertex label:', H.vertex_label())
print('Edge list:', H.edge_list())
# resulting in a new graph with 5 nodes (reduced from 6 nodes)

# look at the spectral feature if k < 5
sf = G.spectral_feature(k=4)
print('\nSpectral feature (k = 4):', sf, sep='\n')

# look at the spectral feature if k = 5
sf = G.spectral_feature(k=5)
print('\nSpectral feature (k = 5): ', sf, sep='\n')

# look at the spectral feature if k > 5
sf = G.spectral_feature(k=8)
print('\nSpectral feature (k = 8): ', sf, sep='\n')
