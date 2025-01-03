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
        # Return the activated input
        self.input = input
        self.output = self.activation(self.input)
        return self.output

    def backward_propagation(self, output_error, learning_rate):
        # Return input_error=de_dX for the given output_error de_dY.
        # no learnable parameters
        return self.activation_prime(self.input) * output_error


class Tanh:
    def __call__(self, X):
        return np.tanh(X)

    def derivative(self, X):
        return 1 - self(X) ** 2


class MSELoss:
    def __call__(self, y_true, y_hat):
        return np.mean((y_true - y_hat) ** 2)

    def derivative(self, y_true, y_hat):
        return 2 * (y_true - y_hat) / y_true.size


class MultiLayerPerceptron:
    def __init__(self):
        self.layers = []
        self.loss = None
        self.loss_derivative = None

    def add_layer(self, layer):
        self.layers.append(layer)

    def set_loss(self, loss, loss_derivative):
        self.loss = loss
        self.loss_derivative = loss_derivative

    def predict(self, input):
        batch_size = input.size(0)  # batch first
        result = []
        for i in range(batch_size):
            output = []
            for layer in self.layers:
                output = layer.forward_propagation(input[i])
            result.append(output)
        return result

    def fit(self, X_train, y_train, learning_rate, epochs):
        batch_size = X_train.shape[0]
        for epoch in range(epochs):
            err = 0
            for i in range(batch_size):
                output = []
                for layer in self.layers:
                    output = layer.forward_propagation(X_train[i])
                err += self.loss(y_train[i], output)
                error = self.loss_derivative(y_train[i], output)
                for layer in reversed(self.layers):
                    error = layer.backward_propagation(error, learning_rate)
            err /= batch_size
            print(f'Epoch {epoch+1}/{epochs}: Error: {err}')


if __name__ == '__main__':
    # training data
    x_train = np.array([[[0, 0]], [[0, 1]], [[1, 0]], [[1, 1]]])
    y_train = np.array([[[0]], [[1]], [[1]], [[0]]])

    # network
    net = MultiLayerPerceptron()
    net.add_layer(FullyConnectedLayer(2, 3))
    net.add_layer(ActivationLayer(tanh, tanh_prime))
    net.add_layer(FullyConnectedLayer(3, 1))
    net.add_layer(ActivationLayer(tanh, tanh_prime))

