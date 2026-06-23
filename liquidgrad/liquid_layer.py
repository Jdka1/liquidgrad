"""
TODO: Add docs.
"""

import numpy as np


class LiquidLayer:
    def __init__(self, n_neurons, speed, seed=None):
        rng = np.random.default_rng(seed)

        self.n_neurons = n_neurons
        self.speed = speed

        self.W = rng.standard_normal((n_neurons, n_neurons)) / np.sqrt(n_neurons)
        self.b = np.zeros(n_neurons)
        self.state = np.zeros(n_neurons)


    def think(self, input):
        next = self.W @ input + self.b
