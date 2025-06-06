from .neuron import Neuron
import numpy as np

class Layer:
    def __init__(self, neurons:list[Neuron]):
        self.neurons = neurons

    def forward(self, x):
        return np.array([n.forward(x) for n in self.neurons])

    def backward(self, grad_output):
        grad_inputs = [n.backward(grad_output[i]) for i, n in enumerate(self.neurons)]
        return np.sum(grad_inputs, axis=0)

    def update(self, learning_rate):
        for n in self.neurons:
            n.update(learning_rate)

    def __str__(self):
        return f"Couche:\n Neurones:{'\n'.join([str(neuron) for neuron in self.neurons])}"

    
