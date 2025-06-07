import numpy as np
import matplotlib.pyplot as plt

def plot_decision_boundary(network, x, targets, resolution=100):
    x_min, x_max = x[:,0].min() - 0.5, x[:,0].max() + 0.5
    y_min, y_max = x[:,1].min() - 0.5, x[:,1].max() + 0.5

    xx, yy = np.meshgrid(np.linspace(x_min, x_max, resolution),
                         np.linspace(y_min, y_max, resolution))
    grid = np.c_[xx.ravel(), yy.ravel()]

    preds = np.array([network.forward(p)[0] for p in grid])
    preds = preds.reshape(xx.shape)

    plt.contourf(xx, yy, preds, levels=50, cmap="bwr", alpha=0.6)

    plt.scatter(x[:, 0], x[:, 1], c=targets[:, 0], cmap="bwr", edgecolor='k')
    plt.title("Frontière de décision")
    plt.axis('equal')