import numpy as np
from functions import Activation_function


class Layer:
    def __init__(self, inputs_n:int, neurons_n:int) :
        self.inputs_n: int = inputs_n
        self.neurons_n: int = neurons_n
        self.weights = np.random.randn(inputs_n, neurons_n)*np.sqrt(2.0/inputs_n)
        self.biases = np.zeros((1, neurons_n))

        self.weight_mask = np.ones((inputs_n, neurons_n))

    @classmethod
    def from_matrix(cls, weights, biases):
        layer = cls(weights.shape[0], weights.shape[1])
        layer.weights = weights
        layer.biases = biases

        layer.weight_mask = np.where(weights != 0, 1.0, 0.0)

        return layer

    def forward(self, inputs):
        self.inputs = inputs

        self.weights *= self.weight_mask

        self.output = np.dot(inputs, self.weights) + self.biases
        return self.output

    def backward(self, dvalues):
        # 1. Jak bardzo zawiniły wagi (dW = X^T * dZ)
        self.dweights = np.dot(self.inputs.T, dvalues)

        # 2. Jak bardzo zawiniły biasy (db = suma dZ)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)

        # 3. Co przekazać warstwie niżej (dX = dZ * W^T)
        self.dinputs = np.dot(dvalues, self.weights.T)

        return self.dinputs