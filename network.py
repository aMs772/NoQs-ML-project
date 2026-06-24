import numpy as np
import random

class Network:

    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y,1) for y in sizes[1:]]
        self.weights = [np.random.rand(y,x) for x,y in zip(sizes[:-1],sizes[1:])]
    
    @staticmethod
    def sigmoid(z):
        return 1.0/(1.0 + np.exp(-z))
    
    @staticmethod
    def sigmoid_prime(z):
        return Network.sigmoid(z)*(1-Network.sigmoid(z))
    
    def feedforward(self, a: np.ndarray) -> np.ndarray:
        # returns output for input a
        for b, w in zip(self.biases, self.weights):
            a = self.sigmoid(w@a + b)
        return a

    def backprop(self, x: np.ndarray, y: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """x is a 784 dimension np.ndarray, this is the activation of initial layer.
           y is a 10 dimension np.ndarray, this is the final expected output.
           returns a tuple of (nabla_b, nabla_w).
           nabla_b, nabla_w are similar to self.biases, self.weights.
           i.e they are lists with each entry being a matrix corresponding to that layer.
        """

        # computing z vectors of each layer (feedforward)
        activation = x
        activations = [x] # for storing activation layer by layer
        zs = []  # for storing z vectors layer by layer
        for b, w in zip(self.biases, self.weights):
            z = w@activation + b
            zs.append(z)
            activation = self.sigmoid(z)
            activations.append(activation)

        # backpropagation
        nabla_b = [np.zeros_like(b) for b in self.biases]
        nabla_w = [np.zeros_like(w) for w in self.weights]
            # nabla_b = list(map(np.zeros_like, self.biases)) also works, but less readable
        delta = self.cost_derivative(activations[-1], y)*self.sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = delta@activations[-2].transpose()
        for l in range(2, self.num_layers):
            z = zs[-l]
            delta = (self.weights[-l+1].transpose()@delta)*self.sigmoid_prime(z)
            nabla_b[-l] = delta
            nabla_w[-l] = delta@activations[-l-1].transpose()
        return (nabla_b, nabla_w)

    def cost_derivative(self, output_activation: np.ndarray, y: np.ndarray) -> np.ndarray:
        """ returns vector of partial derivatives with respect to activations of
            last layer. 
            This is the derivative for quadratic cost_function.
        """
        return (output_activation-y)
    
    def update_mini_batch(self, mini_batch: list[tuple[np.ndarray, np.ndarray]], eta: float):
        """
        updates network parameters.
        mini_batch is a list of tuples
            each tuple consists of 2 arrays x,y.
            x is the input: 784 dimension ndarray
            y is the expected output: 10 dimension array            
        eta is the learning rate
        """

        nabla_b = [np.zeros_like(b) for b in self.biases]
        nabla_w = [np.zeros_like(w) for w in self.weights]
        for x,y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x,y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.biases = [b - (eta/len(mini_batch))*nb for b, nb in zip(self.biases, nabla_b)]
        self.weights = [w - (eta/len(mini_batch))*nw for w, nw in zip(self.weights, nabla_w)]

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""

        test_results=[(np.argmax(self.feedforward(x)), y) for x,y in test_data]
        return sum(x==y for x,y in test_results)

    # stochastic gradient descent
    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        """The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs. If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially.
        """
        
        if test_data: n_test = len(test_data)

        n = len(training_data)
        for i in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+ mini_batch_size] for k in range(0,n,mini_batch_size)
                ]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            
            if test_data:
                print(f"Epoch {i}: {self.evaluate(test_data)} / {n_test}")
            else:
                print(f"Epoch {i} complete")
        


