"""
Student: Mathis Embit
Deep Learning on Graphs - ALTEGRAD - Nov 2023
"""

import networkx as nx
import numpy as np
from scipy.sparse.linalg import eigs
from scipy.sparse import diags, eye

from sklearn.linear_model import LogisticRegression
from sklearn.manifold import SpectralEmbedding
from sklearn.metrics import accuracy_score
from deepwalk import deepwalk
import matplotlib.pyplot as plt

# Loads the karate network
G = nx.read_weighted_edgelist('data/karate.edgelist', delimiter=' ', nodetype=int, create_using=nx.Graph())
print("Number of nodes:", G.number_of_nodes())
print("Number of edges:", G.number_of_edges())

n = G.number_of_nodes()

# Loads the class labels
class_labels = np.loadtxt('data/karate_labels.txt', delimiter=',', dtype=np.int32)
idx_to_class_label = dict()
for i in range(class_labels.shape[0]):
    idx_to_class_label[class_labels[i,0]] = class_labels[i,1]

y = list()
for node in G.nodes():
    y.append(idx_to_class_label[node])

y = np.array(y)


############## Task 5
# Visualizes the karate network

##################
nx.draw_networkx(G, node_color = y)
plt.savefig('Karate.png')
plt.show()
##################


############## Task 6
# Extracts a set of random walks from the karate network and feeds them to the Skipgram model
n_dim = 128
n_walks = 10
walk_length = 20
model = deepwalk(G, n_walks, walk_length, n_dim) # your code here

embeddings = np.zeros((n, n_dim))
for i, node in enumerate(G.nodes()):
    embeddings[i,:] = model.wv[str(node)]

idx = np.random.RandomState(seed=42).permutation(n)
idx_train = idx[:int(0.8*n)]
idx_test = idx[int(0.8*n):]

X_train = embeddings[idx_train,:]
X_test = embeddings[idx_test,:]

y_train = y[idx_train]
y_test = y[idx_test]


############## Task 7
# Trains a logistic regression classifier and use it to make predictions

##################
clf = LogisticRegression(solver='lbfgs')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print('y_pred:', y_pred)
dw_accuracy = accuracy_score(y_test, y_pred)
print('Deepwalk accuracy:', dw_accuracy)
##################


############## Task 8
# Generates spectral embeddings

##################
def eigen_vectors(G, k):
    A = nx.adjacency_matrix(G)
    D_inv = diags([1/G.degree(node) for node in G.nodes()])
    Lrw = eye(G.number_of_nodes()) - D_inv @ A
    evals, evecs = eigs(Lrw, k=k, which = 'SR')
    evecs = np.real(evecs)
    return(evecs)

k = 2
eigen_vectors = eigen_vectors(G, k)
X_train_spec = eigen_vectors[idx_train,:]
X_test_spec = eigen_vectors[idx_test,:]

y_train_spec = y[idx_train]
y_test_spec = y[idx_test]

clf_spec = LogisticRegression().fit(X_train_spec, y_train_spec)
y_pred_spec = clf_spec.predict(X_test_spec)
spec_accuracy = accuracy_score(y_test_spec, y_pred_spec)
print('Spectral accuracy:', spec_accuracy)
##################
