import math
import random

class RBFNetwork(object):
    def __init__(self, node_props, eta):
        self._num_nodes = len(node_props)
        self._eta = eta
        self._weights = [random.uniform(-1, 1) for i in range(self._num_nodes)]
        self._means = [n[0] for n in node_props]
        self._widths = [n[1] for n in node_props]
        self._bias = random.uniform(-1, 1)

    def _gaussians(value):
        return map(lambda m, v: math.exp(-(((m - value)**2) / (2 * v))), self._means, self._widths)

    def feed(self, inp):
        return sum(map(lambda x, y: x * y, self._gaussians(inp), self._weights)) + self._bias

    def train(self, inp, des):
        # Get the result of sending this through
        y_inp = self.feed(inp)

        # Update the weights
        delta_w = map(lambda x: eta * (des - y_inp) * x, self._gaussians(inp))
        self._weights = map(lambda x, y: x + y, self._weights, delta_w)

        # Update the bias
        delta_b = eta * (des - y_inp)
        self._bias = self._bias + delta_b
