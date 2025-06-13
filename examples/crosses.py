import numpy as np
import matplotlib.pyplot as plt
from neural_net import Neuron, Layer, Network, Trainer
from neural_net.activations import LeakyReLU, grad_LeakyReLU, sigmoid, grad_sigmoid
from neural_net.losses import cross_entropy, grad_cross_entropy
from neural_net.neural_saver import save_network
from generate_data.simple_data import generate_cross_data

def plot_crosses(network, x, img_size, file_name):
    ig, axes = plt.subplots(2, 5, figsize=(5, 5))
    for i, ax in enumerate(axes.ravel()):
        ax.imshow(x[i].reshape(img_size, img_size), cmap='gray', vmin=0, vmax=1)
        ax.set_title(f"Label: {round(network.forward(x[i])[0], 2)}")
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(file_name)

def measure_accuracy(network, x, targets):
    sum = 0
    for xi, yi in zip(x, targets):
        sum += int(round(network.forward(xi)[0]) == yi[0])
    return sum/ x.shape[0]

def main():

    img_size = 16

    x_train, targets = generate_cross_data(nb_img= 300, image_size=img_size)

    r = [0.01]
    for i, lr in enumerate(r):
        network = Network([
            Layer([Neuron(nb_inputs=img_size**2, activation=LeakyReLU, grad_activation=grad_LeakyReLU) for _ in range(12)]),
            Layer([Neuron(nb_inputs=12, activation=LeakyReLU, grad_activation=grad_LeakyReLU) for _ in range(6)]),
            Layer([Neuron(nb_inputs=6, activation=sigmoid, grad_activation=grad_sigmoid)])
        ],
        loss=cross_entropy,
        grad_loss=grad_cross_entropy
        )

        trainer = Trainer(network, x_train, targets, learning_rate=lr, nb_epochs=250, epsilon=0.0001, plot=True, verbose=False)
        print(f"lr = {lr}")
        trainer.train()
    save_network("networks/crosses.pkl", network)
        

    plt.savefig("plots/crosses/losses.png")


    x_test, y_test = generate_cross_data(image_size=img_size)
    plot_crosses(network, x_train, img_size, "plots/crosses/crosses_train.png")
    plot_crosses(network, x_test, img_size, "plots/crosses/crosses_test.png")

    print(f"train acc: {measure_accuracy(network, x_train, targets)}")
    print(f"test acc: {measure_accuracy(network, x_test, y_test)}")

if __name__ == "__main__":
    main()