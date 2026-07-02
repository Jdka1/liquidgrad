from liquidgrad import Tensor, Parameter
import numpy as np


class LinearFF:
    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        activation=None,
        seed: int | None = None
    ):
        rng = np.random.default_rng(seed)
        scale = 1.0 / np.sqrt(n_inputs)

        self.n_inputs: int = n_inputs
        self.n_outputs: int = n_outputs
        self.activation = activation

        self.W: Parameter = Parameter(rng.standard_normal((n_inputs, n_outputs)) * scale)
        self.b: Parameter = Parameter(np.zeros(n_outputs))

    def parameters(self) -> list[Parameter]:
        return [self.W, self.b]

    def forward(self, x: Tensor):
        pre = x.data @ self.W.data + self.b.data
        return pre if self.activation is None else self.activation(pre)
    
    def __repr__(self) -> str:
        act = self.activation.__name__ if self.activation else "linear"
        return f"{type(self).__name__}({self.n_inputs}, {self.n_outputs}), act={act}"
    

