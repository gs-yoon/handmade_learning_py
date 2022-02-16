from typing import OrderedDict
import numpy as np
from essential.graph_functions import *
from essential.difference import *

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std = 0.01):
        #init weight
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = Relu()
        self.lastLayer = SoftmaxWithLoss()

    def predict(self, x):
        for layer in self.layers.value():
            x = layer.forward(x)

        return x

    def loss(self, x, t):
        y = self.predict(x)
        return self.lastLayer.forward(y,t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y - np.argmax(y, axis = 1)
        if t.ndim != 1 : t = np.argmax(t, axis = 1)

        accuarcy = np.sum(y == t) / float(x.shape[0])

        return accuarcy

    def numericalGrad(self, x, t):
        loss_W = lambda W: self.loss(x,t)

        grads = {}
        grads['W1'] = numericalGrad(loss_W, self.params['W1'])
        grads['b1'] = numericalGrad(loss_W, self.params['b1'])
        grads['W2'] = numericalGrad(loss_W, self.params['W2'])
        grads['b2'] = numericalGrad(loss_W, self.params['b2'])
        return grads

    def gradient(self, x, t):
        self.loss(x, t)

        dout = 1
        dout = self.lastLayer.backward(dout)

        layers = list(self.layers.values())
        layers.erverse()
        for layer in layers:
            dout = layer.backward(dout)

        grads = {}
        grads['W1'] = self.layers['Affine1'].dW
        grads['b1'] = self.layers['Affine1'].db
        grads['W2'] = self.layers['Affine2'].dW
        grads['b1'] = self.layers['Affine2'].db

        return grads