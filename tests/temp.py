from liquidgrad import Tensor
from liquidgrad.optim import SGD
from liquidgrad.layers import LinearFF
import numpy as np



layer = LinearFF(1,1)
optimizer = SGD(layer.parameters(), lr=0.1)

layer.W.grad += 1.0
layer.b.grad += 1.0

print(layer.W.data, layer.b.data)

optimizer.step()

print(layer.W.data, layer.b.data)
