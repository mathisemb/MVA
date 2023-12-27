"""
Graph Mining - ALTEGRAD - Oct 2023
EMBIT Mathis
"""

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

############## Task 1
filename = 'datasets/CA-HepTh.txt'
G = nx.read_edgelist(filename, comments='#', delimiter='\t')
print('Nb of nodes:', G.number_of_nodes())
print('Nb of deges:', G.number_of_edges())


############## Task 2
print('Nb of connected components:', nx.number_connected_components(G))
largest_cc = max(nx.connected_components(G), key=len)
print('The largest connected component in the graph has', len(largest_cc), ' nodes')
subG = G.subgraph(largest_cc)
print('The largest connected component in the graph has', subG.number_of_edges(), ' edges')
print('Nodes of LCC / nodes of the graph =', len(largest_cc)/len(G))
print('Edges of LCC / edges of the graph =', subG.number_of_edges()/G.number_of_edges())


############## Task 3
# Degree
degree_sequence = [G.degree(node) for node in G.nodes()]
print('Maximum degree of the nodes in the graph:', np.max(degree_sequence))
print('Minimum degree of the nodes in the graph:', np.min(degree_sequence))
print('Median degree of the nodes in the graph:', np.median(degree_sequence))
print('Mean degree of the nodes in the graph:', np.mean(degree_sequence))


############## Task 4
fig, axs = plt.subplots(1, 2)
axs[0].plot(nx.degree_histogram(G))
axs[0].set_xlabel('degree')
axs[0].set_ylabel('frequency')
axs[1].loglog(nx.degree_histogram(G))
axs[1].set_xlabel('degree')
axs[1].set_ylabel('frequency')
plt.show()


############## Task 5
global_clustering = nx.transitivity(G)
print('global_clustering: ', global_clustering)
