import numpy as np

class Neuron:
    def __init__(self, nb_inputs:int, activation, grad_activation):
        self.biais = np.random.randn() * 0.1
        self.weights = np.random.randn(nb_inputs) 
        self.activation = activation
        self.grad_activation = grad_activation
        self.grad_biais = 0
        self.grad_weight = np.zeros(nb_inputs)  
        self.last_input = 0
        self.last_output = 0
        self.last_sum = 0
    
    def somme(self, x):
        return np.dot(self.weights, x) + self.biais

    def forward(self, x):
        self.last_input = x
        self.last_sum = self.somme(x)
        self.last_output = self.activation(self.last_sum)
        return self.last_output

    
    def backward(self, grad_output):
        grad_activ = self.grad_activation(self.last_sum) #da_dz
        delta = grad_output * grad_activ #dl_dz 
        self.grad_biais = delta
        self.grad_weight = delta * self.last_input #dl_dz @ dz_dw
        grad_input = delta * self.weights #dl_dx
        return grad_input


    def update(self, learning_rate:float):
        self.weights -= learning_rate * self.grad_weight
        self.biais -= learning_rate * self.grad_biais
    
    def __str__(self):
        return f"Neurone\nbiais: {self.biais}\nactivation: {self.activation}\n poids: {self.weights}"

