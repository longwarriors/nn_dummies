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
        """Compute dE/dW, dE/dB for the given output_error. Returns input_error.
        input_error=dE/dX
        output_error=dE/dY
        """
        input_error = output_error.dot(self.weights.T)
        weights_error = input_error * self.weights