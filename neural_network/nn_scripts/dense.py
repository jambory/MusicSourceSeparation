import numpy as np
from nn_scripts.layer import Layer

class Dense(Layer):
    def __init__(self, input_size, output_size):
        self.weights= np.random.randn(output_size,input_size) # returns array of size (j,i) sampled randomly from standard normal distribution.
        self.bias=np.random.randn(output_size, 1) # returns array of size (j,1) sampled randomly from standard normal distribution.

    def forward(self, input):
        self.input = input
        return np.dot(self.weights, self.input) + self.bias # performs caclulation from formula above.
    
    def backward(self, output_gradient, learning_rate):
        self.weights_gradient = np.dot(output_gradient, self.input.T)
        self.weights -= learning_rate * self.weights_gradient
        self.bias -= learning_rate * output_gradient
        
        return np.dot(self.weights.T, output_gradient)