import numpy as np
import matplotlib.pyplot as plt
from neural_net import Neuron, Layer, Network, Trainer
from neural_net.activations import identity, grad_identity
from neural_net.losses import MSE, grad_MSE
from neural_net.neural_saver import save_network
from generate_data.simple_data import generate_calc_data

def main():
    x, targets = generate_calc_data(type="sum", norm=False)

    r = [0.00001, 0.001, 0.01]
    plt.figure(figsize=(15, 15))
    for i, lr in enumerate(r):
        layer1 = Layer([Neuron(nb_inputs=2, activation=identity, grad_activation=grad_identity) for _ in range(1)])
        network = Network([layer1], loss = MSE, grad_loss = grad_MSE)

        trainer = Trainer(network, x, targets, learning_rate=lr, nb_epochs=600, epsilon=0.01, plot=True, verbose=False)
        print(f"lr = {lr}")
        plt.subplot(3,3, i+1)
        trainer.train()
        plt.subplots_adjust(wspace=.5, hspace=.3)

        if lr == 0.01:
            save_network(f"networks/sum.pkl", network)
        
        x, targets = generate_calc_data(nb_points=10, type="sum", norm=False)
        for xi, target in zip(x, targets):
            s = network.forward(xi)[0]
            print(f"{xi} = {s} (diff={target-s})")
    plt.savefig('plots/sum/sum.png')

if __name__ == "__main__":
    main()



