import numpy as np
import autograd.numpy as np_
from autograd import grad


class ClassificadorAlfa():

    def __init__(self, learning_rate, iters, params):
        self.learning_rate = learning_rate
        self.iters = iters
        self.params = params
        self.gradient = grad(self.erro, argnum=0)

    def erro(self, params):
        a, b, x, y = params
        yhat = a.T @ x.T + b
        mse = np_.mean((yhat - y)**2)
        return mse

    def melhorar_modelo(self):
        params = self.params
        for i in range(self.iters):
            a, b, x, y = params
            grads = self.gradient( (params) )
            a -= self.learning_rate * grads[0]
            b -= self.learning_rate * grads[1]
            params = [a, b, x, y]
        return params

    def main(self):
        params = self.melhorar_modelo()
        return params
