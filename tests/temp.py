import liquidgrad.layers as layers
import numpy as np

print("\n" * 5)


liquid = layers.Liquid(
    n_neurons=5
)


x = np.array([1,2,3,4,5])
print(x.shape)

print(liquid.state)
for i in range(100):
    liquid.think(x)
print(liquid.state)



print(liquid)



