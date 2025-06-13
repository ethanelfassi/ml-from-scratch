from .neuron import Neuron
import numpy as np

class Layer:
    """
        implementation of a Layer
        (bunch of neurons)
    """
    def __init__(self, neurons:list[Neuron]):
        self.neurons = neurons

    def forward(self, x):
        """
            forwards x for each neuron of the layer
        """
        return np.array([n.forward(x) for n in self.neurons])

    def backward(self, grad_output):
        """
            backwards x for each neuron of the layer
        """
        grad_inputs = [n.backward(grad_output[i]) for i, n in enumerate(self.neurons)]
        return np.sum(grad_inputs, axis=0)

    def update(self, learning_rate):
        """
            updates each neuron of the layer
        """
        for n in self.neurons:
            n.update(learning_rate)

    def __str__(self):
        return f"Couche:\n Neurones:{'\n'.join([str(neuron) for neuron in self.neurons])}"

    
