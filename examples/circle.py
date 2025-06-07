import numpy as np
import matplotlib.pyplot as plt
from neural_net import Neuron, Layer, Network, Trainer
from neural_net.activations import LeakyReLU, grad_LeakyReLU, sigmoid, grad_sigmoid
from neural_net.losses import cross_entropy, grad_cross_entropy
from neural_net.neural_saver import save_network
from generate_data.simple_data import generate_circle_data
from .plot_results import plot_decision_boundary

def plot_boundary_circle(network, x):
    plt.scatter(x[:, 0], x[:,1], c=[network.forward(xi)[0] for xi in x], cmap='ocean', edgecolor='k')
    plt.colorbar(label="Sortie du réseau")
    circle = plt.Circle((0, 0), radius=0.5, edgecolor='red', facecolor='none', linewidth=2)
    plt.gca().add_patch(circle)
    plt.gca().set_aspect('equal')

def main():
    x_train, targets = generate_circle_data(1000)

    r = [0.01]
    for i, lr in enumerate(r):
        network = Network([
            Layer(neurons=[Neuron(nb_inputs=2, activation=LeakyReLU, grad_activation=grad_LeakyReLU) for _ in range(6)]),
            Layer(neurons=[Neuron(nb_inputs=6, activation=sigmoid, grad_activation=grad_sigmoid)])
        ],
        loss=cross_entropy,
        grad_loss=grad_cross_entropy
        )

        trainer = Trainer(network, x_train, targets, learning_rate=lr, nb_epochs=1000, epsilon=0.01, plot=True, verbose=False)
        print(f"lr = {lr}")
        trainer.train()
        save_network("networks/circle.pkl", network)
        

    plt.savefig('plots/circle/losses.png')
    plt.figure()

    plot_boundary_circle(network, x_train)

    plt.savefig('plots/circle/circle_train.png')
    plt.figure()
    plot_decision_boundary(network, x_train, targets)
    plt.savefig('plots/circle/boundary_train.png')
    plt.figure()

    x_test, y_test = generate_circle_data(10000)

    plot_boundary_circle(network, x_test)
    plt.savefig('plots/circle/circle_test.png')
    plt.figure()
    plot_decision_boundary(network, x_test, y_test)
    plt.savefig('plots/circle/boundary_test.png')
    plt.figure()

if __name__ == "__main__":
    main()