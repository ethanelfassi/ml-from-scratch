import numpy as np

def MSE(output, target):
    return np.mean((target - output)**2)

def grad_MSE(output, target):
    return -2 * (target - output) / output.size

def MAE(output, target):
    return np.mean(np.abs(target - output))

def grad_MAE(output, target):
    return np.sign(output - target) / output.size

def cross_entropy(output, target):
    epsilon = 1e-15
    output = np.clip(output, epsilon, 1 - epsilon)
    return -np.mean(target * np.log(output) + (1 - target) * np.log(1 - output))

def grad_cross_entropy(output, target):
    epsilon = 1e-15
    output = np.clip(output, epsilon, 1 - epsilon)
    return - (target / output) + ((1 - target) / (1 - output))

def categorical_cross_entropy(output, target):
    epsilon = 1e-15
    return -np.sum(target * np.log(output+epsilon))

def grad_categorical_cross_entropy(output, target):
    return output - target