"""
Student: Mathis Embit
Deep Learning on Graphs - ALTEGRAD - Nov 2023
"""

import networkx as nx
import numpy as np
import torch
from random import randint

def create_dataset(nb_graphs = 50, probas = [0.2, 0.4]):
    Gs = list() # graph list
    y = list() # labels

    ############## Task 1
    
    ##################
    for label, p in enumerate(probas):
        for _ in range(nb_graphs):
            n = randint(10, 20)
            Gs.append(nx.fast_gnp_random_graph(n, p=p))
            y.append(label)
    ##################

    return Gs, y

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(indices, values, shape)

Gs, y = create_dataset()
#print(Gs)
print(Gs[0])
