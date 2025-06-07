import numpy as np
import matplotlib.pyplot as plt
from neural_net import Neuron, Layer, Network, Trainer
from neural_net.activations import LeakyReLU, grad_LeakyReLU, sigmoid, grad_sigmoid
from neural_net.losses import cross_entropy, grad_cross_entropy
from neural_net.neural_saver import save_network
from generate_data.simple_data import xor_data

def main():
    x, targets = xor_data()

    r = [0.00001, 0.001, 0.01, 0.1, 1, 10, 100]
    plt.figure(figsize=(15, 15))
    for i, lr in enumerate(r):
        layer1 = Layer([Neuron(nb_inputs=2, activation=LeakyReLU, grad_activation=grad_LeakyReLU) for _ in range(4)])
        layer2 = Layer([Neuron(nb_inputs=4, activation=sigmoid, grad_activation=grad_sigmoid)])
        network = Network([layer1, layer2], loss = cross_entropy, grad_loss = grad_cross_entropy)

        trainer = Trainer(network, x, targets, learning_rate=lr, nb_epochs=10000, epsilon=0.01, plot=True, verbose=False)
        print(f"lr = {lr}")
        plt.subplot(3,3, i+1)
        trainer.train()
        plt.subplots_adjust(wspace=.5, hspace=.3)

        if lr == 0.01:
            save_network(f"networks/xor.pkl", network)
    plt.savefig('plots/xor/xor.png')

if __name__ == "__main__":
    main()



