"""
TODO: Add docs.
"""

import numpy as np


class Liquid:
    def __init__(self, n_neurons, seed=None):
        rng = np.random.default_rng(seed)
        scale = 1.0 / np.sqrt(n_neurons)

        self.n_neurons = n_neurons

        self.W = rng.standard_normal((n_neurons, n_neurons)) * scale
        self.R = rng.standard_normal((n_neurons, n_neurons)) * scale
        self.b = np.zeros(n_neurons)
        self.state = np.zeros(n_neurons)

    def think(self, x):
        input = self.W @ x
        thinking = self.R @ self.state
        self.state += input + thinking + self.b
        return self.state    

    def __repr__(self):
        return f"{type(self).__name__}({self.n_neurons})"