import numpy as np
import matplotlib.pyplot as plt
from neural_net import Neuron, Layer, NetworkMultiClasses, Trainer
from neural_net.activations import LeakyReLU, grad_LeakyReLU, identity, grad_identity
from neural_net.losses import categorical_cross_entropy, grad_categorical_cross_entropy
from neural_net.neural_saver import save_network
from generate_data.simple_data import generate_patterns_data

def plot_crosses_multi_classes(network, x, img_size, file_name):
    ig, axes = plt.subplots(2, 5, figsize=(5, 5))
    for i, ax in enumerate(axes.ravel()):
        ax.imshow(x[i].reshape(img_size, img_size), cmap='gray', vmin=0, vmax=1)
        probas = network.forward(x[i])
        pred = np.argmax(probas)
        ax.set_title(f"Pred: {pred}")
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(file_name)

def measure_accuracy_multi_classes(network, x, targets):
    sum = 0
    for xi, yi in zip(x, targets):
        pred = np.argmax(network.forward(xi))
        true = np.argmax(yi)  
        sum += int(pred == true)
    return sum/ x.shape[0]

def transform_targets(targets, nb_classes):
    res = []
    for target in targets:
        res.append([0 if target!=i else 1 for i in range(nb_classes)])
    return res


def main():

    img_size = 14
    nb_classes = 4
    x_train, targets = generate_patterns_data(nb_img= 1000, image_size=img_size, n_labels=nb_classes)
    targets_train = transform_targets(targets, nb_classes)

    r = [0.01]
    for i, lr in enumerate(r):
        network = NetworkMultiClasses([
            Layer([Neuron(nb_inputs=img_size**2, activation=LeakyReLU, grad_activation=grad_LeakyReLU) for _ in range(12)]),
            Layer([Neuron(nb_inputs=12, activation=LeakyReLU, grad_activation=grad_LeakyReLU) for _ in range(6)]),
            Layer([Neuron(nb_inputs=6, activation=identity, grad_activation=grad_identity) for _ in range(nb_classes)])
        ],
        loss=categorical_cross_entropy,
        grad_loss=grad_categorical_cross_entropy
        )

        trainer = Trainer(network, x_train, targets_train, learning_rate=lr, nb_epochs=400, epsilon=0.0001, plot=True, verbose=False)
        print(f"lr = {lr}")
        trainer.train()
    save_network("networks/patterns.pkl", network)

    plt.savefig("plots/patterns/losses.png")


    x_test, y_test = generate_patterns_data(image_size=img_size, n_labels=nb_classes)
    plot_crosses_multi_classes(network, x_train, img_size, "plots/patterns/patterns_train.png")
    plot_crosses_multi_classes(network, x_test, img_size, "plots/patterns/patterns_test.png")


    targets_test = transform_targets(y_test, nb_classes)
    print(f"train acc: {measure_accuracy_multi_classes(network, x_train, targets_train)}")
    print(f"test acc: {measure_accuracy_multi_classes(network, x_test, targets_test)}")



if __name__ == "__main__":
    main()