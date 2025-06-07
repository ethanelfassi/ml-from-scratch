import numpy as np

def xor_data():
    x = np.array([[0,0], [0,1], [1,0], [1, 1]])
    targets = np.array([[0], [1], [1], [0]])
    return x, targets

def generate_circle_data(nb_points=1000, radius=0.5):
    x = np.random.uniform(-1, 1, (nb_points, 2))
    targets = (np.sum(x**2, axis=1) < radius**2).astype(int).reshape(-1, 1)
    return x, targets

def generate_spiral_data(nb_points=1000, noise=0.1):
    n = nb_points // 2
    theta = np.sqrt(np.random.rand(n)) * 4 * np.pi
    r = theta

    # Spirale 1 (classe 0)
    x1 = np.array([r * np.cos(theta), r * np.sin(theta)]).T
    x1 += np.random.randn(n, 2) * noise
    y1 = np.zeros((n, 1))

    # Spirale 2 (classe 1) : opposée
    x2 = np.array([-r * np.cos(theta), -r * np.sin(theta)]).T
    x2 += np.random.randn(n, 2) * noise
    y2 = np.ones((n, 1))

    x = np.vstack((x1, x2))
    targets = np.vstack((y1, y2))
    return x, targets