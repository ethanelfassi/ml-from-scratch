from .layer import Layer

class Network:
    """
        implementation of a network
        (multiple layers + loss function and gradient of the loss func)
    """
    def __init__(self, layers:list[Layer], loss, grad_loss):
        self.layers = layers
        self.loss = loss
        self.grad_loss = grad_loss

    def forward(self, x):
        """
            forwards x for each layer of the network
            (from first to last)
        """
        for layer in self.layers:
            x = layer.forward(x)
        return x
            
    def backward(self, output, target):
        """
            backwards x for each layer of the network
            (from last to first)
        """
        grad_loss = self.grad_loss(output, target)
        grad_input = grad_loss
        for layer in self.layers[::-1]:
            grad_input = layer.backward(grad_input)

    def loss(self, output, target):
        """
            returns the loss for an output (forward) and a target
        """
        return self.loss(output, target)

    def update(self, learning_rate):
        """
            updates each layer of the network
        """
        for layer in self.layers:
            layer.update(learning_rate)
    
    def __str__(self):
        return f"Reseau:\n Couches:{'\n'.join([str(layer) for layer in self.layers])}"
