import numpy as np

def identity(x):
    return x

def grad_identity(x):
    return 1

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

def tanh(x):
    return (np.exp(x) - np.exp(-x))/((np.exp(x) + np.exp(-x)))

def grad_tanh(x):
    return 1 - tanh(x)**2

def tanh_norm(x):
    return (tanh(x) + 1)/2

def grad_tanh_norm(x):
    return grad_tanh(x)/2

def softmax(z):
    z = np.array(z)
    z_stable = z - np.max(z)
    exp_z = np.exp(z_stable)
    s = exp_z / np.sum(exp_z)
    return s

def grad_softmax(s):
    s = s.reshape(-1, 1)
    res = np.diagflat(s) - np.dot(s, s.T)
    return res[0][0]
