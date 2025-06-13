from .network import Network
from .activations import softmax

class NetworkMultiClasses(Network):
    def __init__(self, layers, loss, grad_loss):
        super().__init__(layers, loss, grad_loss)
    
    def forward(self, x):
        return softmax(super().forward(x))