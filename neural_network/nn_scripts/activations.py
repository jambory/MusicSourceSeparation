import numpy as np
from nn_scripts.layer import Layer

class Activation(Layer):
    def __init__(self, activation, activation_prime):
        self.activation=activation # this is a function.
        self.activation_prime=activation_prime # also a function.
    def forward(self, input):
        self.input = input
        return self.activation(self.input) # applies inputs to activation function and returns outputs.
    def backward(self, output_gradient, learning_rate):
        return np.multiply(output_gradient, self.activation_prime(self.input))
    
class Tanh(Activation):
    def __init__(self):
        tanh = lambda x: np.tanh(x)
        tanh_prime = lambda x: 1 - np.tanh(x) ** 2
        super().__init__(tanh, tanh_prime)

class Sigmoid(Activation):
    def __init__(self):
        sigmoid = lambda x: 1 / (1 + np.exp(-x))
        def sigmoid_prime(x):
            s = sigmoid(x)
            return s * (1 - s)
        super().__init__(sigmoid, sigmoid_prime)
        
class ReLu(Activation):
    def __init__(self):
        relu = lambda x: np.max(0,x)
        relu_prime = lambda x: 1 if x >= 0 else 0
        super().__init__(relu, relu_prime)