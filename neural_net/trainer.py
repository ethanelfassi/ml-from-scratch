from .network import Network
import numpy as np
import matplotlib.pyplot as plt

class Trainer:
    def __init__(self, network:Network, x:np.array, targets:np.array, learning_rate:float,
                nb_epochs:int, epsilon:int, plot:bool=True, verbose:bool=True):
        self.network = network
        self.x = x
        self.targets = targets
        self.learning_rate = learning_rate
        self.nb_epochs = nb_epochs
        self.epsilon = epsilon
        self.plot = plot
        self.verbose = verbose
    
    def train(self):
        losses = []
        for epoch in range(self.nb_epochs):
            total_loss = 0
            outputs = []
            for xi, target in zip(self.x, self.targets):
                output = self.network.forward(xi)
                self.network.backward(output, target)
                self.network.update(self.learning_rate)
                total_loss += self.network.loss(output, target)
                outputs.append(output[0])
            loss_epoch = total_loss/len(self.x)
            losses.append(loss_epoch)

            if self.verbose:
                print(f"Epoch {epoch+1} - outputs: {outputs} - Loss: {total_loss / len(self.x)}")

            if loss_epoch < self.epsilon:
                break

               
        
        if self.verbose:
            print(f"targets: {self.targets}")
        
        if self.plot:
            plt.plot(losses)
            plt.xlabel("Époque")
            plt.ylabel(f"Perte")
            plt.title(f"lr={self.learning_rate}")
            plt.grid(True)
            plt.ylim(0, max(1.5, max(losses)))