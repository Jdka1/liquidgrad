import liquidgrad.liquid_layer

liquid_net = liquidgrad.liquid_layer.LiquidLayer(
    n_neurons=5,
    speed=0.5,
)

print(liquid_net.state)