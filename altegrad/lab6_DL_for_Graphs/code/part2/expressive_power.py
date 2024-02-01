"""
Student: Mathis Embit
Deep Learning on Graphs - ALTEGRAD - Nov 2023
"""

import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch

from models import GNN
from utils import sparse_mx_to_torch_sparse_tensor

# Initializes device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Hyperparameters
hidden_dim = 32
output_dim = 4
dropout = 0.0
neighbor_aggr = 'mean'
readout = 'mean'


############## Task 4
        
##################
dataset = [nx.cycle_graph(n) for n in range(10, 20)]
##################


############## Task 5
        
##################
adj_matrices = [nx.adjacency_matrix(g) for g in dataset]
adj_block_diag = sp.block_diag(adj_matrices)

idx = [np.ones(g.number_of_nodes())*id for id, g in enumerate(dataset)]
idx = np.concatenate(idx)
idx = torch.LongTensor(idx).to(device)

# features
x = np.ones((adj_block_diag.shape[0],1))
x = torch.FloatTensor(x).to(device)
adj_block_diag = sparse_mx_to_torch_sparse_tensor(adj_block_diag).to(device)
##################


############## Task 8
        
##################
input_dim = 1
combinations = [('mean','mean'), ('mean','sum'), ('sum','mean'), ('sum','sum')]
for neighbor_aggr, readout in combinations:
    with torch.no_grad():
        model = GNN(input_dim, hidden_dim, output_dim, neighbor_aggr, readout, dropout).to(device)
        representation = model(x, adj_block_diag, idx)
    print("neighbor_aggr =", neighbor_aggr, "and readout =", readout)
    print("representation:\n", representation.detach().cpu().numpy(), "\n")
##################


############## Task 9
        
##################
G1 = nx.union(nx.cycle_graph(3), nx.cycle_graph(3), rename=('C3a', 'C3b'))
G2 = nx.cycle_graph(6)
##################


############## Task 10
        
##################
dataset = [G1, G2]

adj_matrices = [nx.adjacency_matrix(g) for g in dataset]
adj_block_diag = sp.block_diag(adj_matrices)

idx = [np.ones(g.number_of_nodes())*id for id, g in enumerate(dataset)]
idx = np.concatenate(idx)
idx = torch.LongTensor(idx).to(device)

# features
x = np.ones((adj_block_diag.shape[0],1))
x = torch.FloatTensor(x).to(device)
adj_block_diag = sparse_mx_to_torch_sparse_tensor(adj_block_diag).to(device)
##################


############## Task 11
        
##################
with torch.no_grad():
    model = GNN(input_dim, hidden_dim, output_dim, 'sum', 'sum', dropout).to(device)
    representation = model(x, adj_block_diag, idx)
representation = representation.detach().cpu().numpy()
print("G1 representation:", representation[0, :])
print("G2 representation:", representation[1, :])
##################


############## Question 4
'''
G1 = nx.union(nx.path_graph(3), nx.path_graph(3), rename=('P3a', 'P3b'))
G1 = nx.union(G1, nx.path_graph(3), rename=('P3aP3b', 'P3c'))

G2 = nx.union(nx.cycle_graph(3), nx.path_graph(2), rename=('C3', 'P2a'))
G2 = nx.union(G2, nx.path_graph(2), rename=('C3P2a', 'P2b'))
G2 = nx.union(G2, nx.path_graph(2), rename=('C3P2aP2b', 'P2c'))
'''

G1 = nx.union(nx.cycle_graph(3), nx.cycle_graph(5), rename=('C3', 'C5'))
G2 = nx.cycle_graph(8)

dataset = [G1, G2]

adj_matrices = [nx.adjacency_matrix(g) for g in dataset]
adj_block_diag = sp.block_diag(adj_matrices)

idx = [np.ones(g.number_of_nodes())*id for id, g in enumerate(dataset)]
idx = np.concatenate(idx)
idx = torch.LongTensor(idx).to(device)

x = np.ones((adj_block_diag.shape[0],1))
x = torch.FloatTensor(x).to(device)
adj_block_diag = sparse_mx_to_torch_sparse_tensor(adj_block_diag).to(device)

with torch.no_grad():
    model = GNN(input_dim, hidden_dim, output_dim, 'sum', 'sum', dropout).to(device)
    representation = model(x, adj_block_diag, idx)
representation = representation.detach().cpu().numpy()
print("\nG1 representation:", representation[0, :])
print("G2 representation:", representation[1, :])
