import numpy as np
import matplotlib.pyplot as plt
from neural_net import Neuron, Layer, Network, Trainer
from neural_net.activations import identity, grad_identity, LeakyReLU, grad_LeakyReLU, tanh, tanh_norm
from neural_net.losses import MSE, grad_MSE, cross_entropy, grad_cross_entropy, MAE, grad_MAE
from neural_net.neural_saver import save_network
from generate_data.simple_data import generate_calc_data

def main():
    x, targets = generate_calc_data(type="product", norm = False)

    r = [0.00001, 0.0001, 0.001]
    plt.figure(figsize=(15, 15))
    for i, lr in enumerate(r):
        layer1 = Layer([Neuron(nb_inputs=2, activation=tanh, grad_activation=tanh_norm) for _ in range(8)])
        layer2 = Layer([Neuron(nb_inputs=8, activation=identity, grad_activation=grad_identity) for _ in range(1)])
        network = Network([layer1, layer2], loss = MSE, grad_loss = grad_MSE)

        trainer = Trainer(network, x, targets, learning_rate=lr, nb_epochs=800, epsilon=0.01, plot=True, verbose=True)
        print(f"lr = {lr}")
        plt.subplot(3,3, i+1)
        trainer.train()
        plt.subplots_adjust(wspace=.5, hspace=.3)

        if lr == 0.01:
            save_network(f"networks/product.pkl", network)

        x, targets = generate_calc_data(nb_points=10, type="product", norm=False)
        for xi, target in zip(x, targets):
            s = network.forward(xi)[0]
            print(f"{xi} = {s} (diff={target-s})")


    plt.savefig('plots/product/product.png')

if __name__ == "__main__":
    main()



