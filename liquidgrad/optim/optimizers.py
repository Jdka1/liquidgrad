from liquidgrad.tensor import Parameter


class SGD:
    def __init__(self, parameters: list[Parameter], lr: float = 0.01):
        self.parameters: list[Parameter] = parameters
        self.lr: float = lr
        
    def step(self):
        for parameter in self.parameters:
            parameter.data -= self.lr * parameter.grad
            
    def zero_grad(self):
        for parameter in self.parameters:
            parameter.grad.fill(0.0)