import numpy as np

class Network:

    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y,1) for y in sizes[1:]]
        self.weights = [np.random.rand(y,x) for y,x in zip(sizes[:-1],sizes[1:])]
    
    @staticmethod
    def sigmoid(z):
        return 1.0/(1.0 + np.exp(-z))
    
    def feedforward(self, a):
        # returns output for input a
        for b, w in zip(self.biases, self.weights):
            a = self.sigmoid(w@a + b)
        return a

    # stochastic gradient descent
    