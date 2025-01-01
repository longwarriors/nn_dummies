# https://towardsdatascience.com/math-neural-network-from-scratch-in-python-d6da9f29ce65
import numpy as np


class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    def forward_propagation(self, input):
        """compute the output Y of the layer for a given input X"""
        raise NotImplementedError

    def backward_propagation(self, output_error, learning_rate):
        """compute the dE/dX for the given dE/dY and update the weights if any"""
        raise NotImplementedError


class FullyConnectedLayer(Layer):
    # input_size = number of input neurons
    # output_size = number of output neurons
    def __init__(self, input_size, output_size):
        self.weights = np.random.rand(input_size, output_size) - 0.5
        self.bias = np.random.rand(1, output_size) - 0.5

    def forward_propagation(self, input):
        self.input = input
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output

    def backward_propagation(self, output_error, learning_rate):
        """Compute dE/dW, dE/dB for the given output_error.
        Returns input_error.
        input_error=dE/dX
        output_error=dE/dY
        """
        input_error = output_error.dot(self.weights.T)  # de_dX = de_dY @ W.T
        weights_error = self.input.T.dot(output_error)  # de_dW = X.T @ de_dY
        bias_error = output_error  # de_dB = de_dY

        # update parameters
        self.weights -= learning_rate * weights_error
        self.bias -= learning_rate * bias_error
        return input_error  # de_dX


class ActivationLayer(Layer):
    def __init__(self, activation, activation_derivative):
        self.activation = activation
        self.activation_prime = activation_derivative

    def forward_propagation(self, input):
