"""
Graph Mining - ALTEGRAD - Oct 2023
EMBIT Mathis
"""

import networkx as nx
import numpy as np
from scipy.sparse.linalg import eigs
from scipy.sparse import diags, eye
from random import randint
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

############## Task 6
# Perform spectral clustering to partition graph G into k clusters
def spectral_clustering(G, k):
    # G: graph
    # k: nb of clusters
    A = nx.adjacency_matrix(G)
    D_inv = diags([1/G.degree(node) for node in G.nodes()])
    Lrw = eye(G.number_of_nodes()) - D_inv@A
    evals, evecs = eigs(Lrw, k=k, which='SR')
    evecs = np.real(evecs)
    kmeans = KMeans(n_clusters=k).fit(evecs)
    clustering = {}
    for i, node in enumerate(G.nodes()):
        clustering[node] = kmeans.labels_[i]
    return clustering


############## Task 7
filename = 'datasets/CA-HepTh.txt'
G = nx.read_edgelist(filename, comments='#', delimiter='\t')
largest_cc = G.subgraph(max(nx.connected_components(G), key=len))
k = 50
clusters = spectral_clustering(largest_cc, k)
# Count the number of keys in each cluster:
cluster_counts = {}
for key, cluster in clusters.items():
    cluster_counts[cluster] = cluster_counts.get(cluster, 0) + 1
# Extract x and y values for the histogram:
clusters_list, counts = zip(*cluster_counts.items())
# Plotting the histogram:
plt.bar(clusters_list, counts, color='skyblue')
plt.xlabel('Clusters')
plt.ylabel('Number of keys')
plt.title('Histogram of keys in each cluster')
plt.show()


############## Task 8
# auxiliary function
def values_to_clusters_dict(original_dict):
    clusters_dict = {}
    # Iterate through the original dictionary
    for value, cluster in original_dict.items():
        # If the cluster is not already a key in clusters_dict, add it:
        if cluster not in clusters_dict:
            clusters_dict[cluster] = []
        # Append the value to the corresponding cluster:
        clusters_dict[cluster].append(value)
    return clusters_dict

# Compute modularity value from graph G based on clustering
def modularity(G, clustering):
    # G: graph
    # clustering: dictionary (keys: id, value: cluster the id belongs to)
    clusters = values_to_clusters_dict(clustering)
    m = nx.number_of_edges(G)
    Q = 0
    for cluster, nodes in clusters.items():
        community = nx.subgraph(G, nodes)
        lc = nx.number_of_edges(community)
        dc = 0 # sum of the degrees of the nodes that belong to community
        for node in community.nodes():
            dc += nx.degree(G, node)
        Q += (lc/m) - (dc/(2*m))**2
    return Q


############## Task 9
# (i) the one obtained by the Spectral Clustering algorithm using k = 50
Q1 = modularity(largest_cc, clusters) # clusters = spectral_clustering(largest_cc, 50) previously computed
print('The modularity of spectral_clustering(largest_cc, 50) is', Q1)
# (ii) the one obtained if we randomly partition the nodes into 50 clusters
rd_clustering = {}
for i, node in enumerate(G.nodes()):
    rd_clustering[node] = randint(1,k)
Q2 = modularity(largest_cc, rd_clustering)
print('The modularity of random clustering is', Q2)
