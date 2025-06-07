import numpy as np
import matplotlib.pyplot as plt
from neural_net import Neuron, Layer, Network, Trainer
from neural_net.activations import LeakyReLU, grad_LeakyReLU, sigmoid, grad_sigmoid
from neural_net.losses import cross_entropy, grad_cross_entropy
from neural_net.neural_saver import save_network
from generate_data.simple_data import generate_spiral_data
from .plot_results import plot_decision_boundary


def plot_boundary(network, x):
    plt.scatter(x[:, 0], x[:,1], c=[network.forward(xi)[0] for xi in x], cmap='ocean', edgecolor='k')
    plt.colorbar(label="Sortie du réseau")
    plt.gca().set_aspect('equal')

def main():
    x_train, targets = generate_spiral_data(nb_points= 1700, noise=0.15)

    r = [0.01]
    for i, lr in enumerate(r):
        network = Network([
            Layer([Neuron(2, LeakyReLU, grad_LeakyReLU) for _ in range(8)]),
            Layer([Neuron(8, LeakyReLU, grad_LeakyReLU) for _ in range(8)]),
            Layer([Neuron(8, sigmoid, grad_sigmoid)])
        ],
        loss=cross_entropy,
        grad_loss=grad_cross_entropy
        )

        trainer = Trainer(network, x_train, targets, learning_rate=lr, nb_epochs=1000, epsilon=0.0001, plot=True, verbose=False)
        print(f"lr = {lr}")
        trainer.train()
        save_network("networks/spirals.pkl", network)

    plt.savefig('plots/spirals/losses.png')
    plt.figure()
    plot_boundary(network, x_train)
    plt.savefig('plots/spirals/spirals_train.png')
    plt.figure()

    plot_decision_boundary(network, x_train, targets)
    plt.savefig('plots/spirals/boundary_train.png')
    plt.figure()

    x_test, y_test = generate_spiral_data(nb_points=3000, noise=0.5)

    plot_boundary(network, x_test)
    plt.savefig('plots/spirals/spirals_test.png')
    plt.figure()
    
    plot_decision_boundary(network, x_test, y_test)
    plt.savefig('plots/spirals/boundary_test.png')
    plt.figure()

if __name__ == "__main__":
    main()