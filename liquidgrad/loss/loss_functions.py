from abc import ABC
import numpy as np
from liquidgrad import Tensor


class LossFunction(ABC):
    @abstractmethod
    def __call__(self, pred: Tensor, target: Tensor) -> Tensor:
        pass
    

class MSELoss(LossFunction):
    def __call__(self, pred: Tensor, target: Tensor) -> Tensor:
        diff = pred.data - target.data
        loss = Tensor((diff ** 2).mean())
        
        def _backward():
            pred.grad