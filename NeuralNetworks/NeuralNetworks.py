import numpy as np
from math import exp

class NeuralNetworks:
    @staticmethod
    def read_data(filename):
        data = np.genfromtxt(filename, delimiter=',')
        outputs = data[:, -1]
        data = data[:, :-1]
        return data, outputs
    
    def __init__(self, training_file, testing_file, layer_sizes):
        self.train_data = NeuralNetworks.read_data(training_file)
        self.test_data = NeuralNetworks.read_data(testing_file)

    @staticmethod
    def loss_backpropagation(x, actual_y, weights, just_forward=False):
        # forward pass
        cache_history = []
        hidden_layers = len(weights) - 1
        for l in range(hidden_layers):
            x, cache = NeuralNetworks.linear_sigmoid_forward(x, weights[l])
            cache_history.append(cache)
        y, last_cache = NeuralNetworks.linear_forward(x, weights[-1])
        if just_forward:
            return y
        
        weight_dev = []
        dy = NeuralNetworks.loss_backward(y, actual_y)
        dx, dw = NeuralNetworks.linear_backward(dy, last_cache)
        dx = np.squeeze(dx)
        weight_dev.insert(0, dw)
        for l in reversed(range(hidden_layers)):
            dx, dw = NeuralNetworks.linear_sigmoid_backward(dx, cache_history[l])
            weight_dev.insert(0, dw)
        return y, weight_dev

    @staticmethod
    def linear_forward(x, w):
        x = np.append(1, x)
        out = x @ w
        cache = (x, w)
        return out, cache

    @staticmethod
    def linear_backward(dout, cache):
        pass

    @staticmethod
    def sigmoid(s):
        return 1/(1+np.exp(-s))

    @staticmethod
    def sigmoid_forward(s):
        cache = s
        return NeuralNetworks.sigmoid(s), cache

    @staticmethod
    def sigmoid_backward(dout, cache):
        pass
    
    @staticmethod
    def linear_sigmoid_forward(x, weights):
        output, lin_cache = NeuralNetworks.linear_forward(x, weights)
        output, sg_cache = NeuralNetworks.sigmoid_forward(output)
        cache = (lin_cache, sg_cache)
        return output, cache
    
    @staticmethod
    def linear_sigmoid_backward(dout, cache):
        pass

    @staticmethod
    def loss_backward(y, ystar):
        pass