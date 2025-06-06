import numpy as np

def ReLU(x):
    return max(0, x)

def grad_ReLU(x):
    return float(x>0)

def LeakyReLU(x):
    return x if x > 0 else 0.01 * x

def grad_LeakyReLU(x):
    return 1.0 if x > 0 else 0.01

def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1/(1+np.exp(-x))

def grad_sigmoid(x):
    y = sigmoid(x)
    return y * (1 - y)
