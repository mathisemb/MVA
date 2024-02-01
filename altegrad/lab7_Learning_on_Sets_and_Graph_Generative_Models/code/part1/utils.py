"""
Student: Mathis Embit
Learning on Sets and Graph Generative Models - ALTEGRAD - Nov 2023
"""

import numpy as np


def create_train_dataset():
    n_train = 100000
    max_train_card = 10

    ############## Task 1
    
    ##################
    X_train = np.zeros((n_train, max_train_card))
    y_train = np.zeros(n_train)
    cards = np.random.randint(1, max_train_card + 1, size=n_train)
    rand_vals = np.random.randint(1, max_train_card + 1, size=(n_train, max_train_card))
    for i, card in enumerate(cards):
        X_train[i, -card:] = rand_vals[i, :card]
        y_train[i] = np.sum(X_train[i, :])
    ##################

    return X_train, y_train


def create_test_dataset():
    
    ############## Task 2
    
    ##################
    n_test=200000
    min_card = 5
    max_card = 100
    card_step = 5
    cards = range(min_card, max_card+1, card_step)
    n_samples_per_card = n_test // len(cards)
    X_test = []
    y_test = []
    for card in cards:
        x = np.random.randint(1, 11, size=(n_samples_per_card, card))
        y = x.sum(axis=1)
        X_test.append(x)
        y_test.append(y)
    ##################

    return X_test, y_test
