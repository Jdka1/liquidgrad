import numpy as np


class LinearFF:
    def __init__(self, n_inputs, n_outputs, activation=None, seed=None):
        rng = np.random.default_rng(seed)
        scale = 1.0 / np.sqrt(n_inputs)

        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.activation = activation

        self.W = rng.standard_normal((n_inputs, n_outputs)) * scale
        self.b = np.zeros(n_outputs)

    def forward(self, x):
        pre = self.W @ x + self.b
        return pre if self.activation is None else self.activation(pre)
    
    def __repr__(self):
        act = self.activation.__name__ if self.activation else "linear"
        return f"{type(self).__name__}({self.n_inputs}, {self.n_outputs}), act={act}"
    

