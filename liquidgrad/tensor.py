import numpy as np


class Tensor:
    def __init__(self, data: np.ndarray | list):
        self.data: np.ndarray = np.array(data, dtype=float)
        self.grad: np.ndarray = np.zeros_like(self.data)
        
    def __repr__(self) -> str:
        return f"{type(self).__name__}(data={self.data}, grad={self.grad})"
    

class Parameter(Tensor):
    pass