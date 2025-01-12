import random
from micrograd.node import Node

class Module:
    def zero_grad(self)->None:
        for p in self.parameters():
            p.gradient = 0

    def parameters(self):
        return []


class Neuron(Module):

    def __init__(self, num_in, nonlin):
        self.weights = [Node(random.uniform(-1,1)) for _ in range(num_in)]
        self.bias = Node(0)
        self.nonlin = nonlin

    def __call__(self, x):
        act = sum((wi*xi for wi,xi in zip(self.weights, x)), self.bias)
        return act.relu() if self.nonlin else act

    def parameters(self):
        return self.weights + [self.bias]

    def __repr__(self):
        return f"{'ReLU' if self.nonlin else 'Linear'}Neuron({len(self.weights)})"

class Layer(Module):

    def __init__(self, num_in, num_out, **kwargs):
        self.neurons = [Neuron(num_in, **kwargs) for _ in range(num_out)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"

class MLP(Module):

    def __init__(self, num_in, num_out):
        sz = [num_in] + num_out
        self.layers = [Layer(sz[i], sz[i+1], nonlin=i!=len(num_out)-1) for i in range(len(num_out))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"