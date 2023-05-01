import numpy as np
import autograd.numpy as np_
from autograd import grad
from sklearn.metrics import accuracy_score


class ClassificadorAlfa():

    def __init__(self, learning_rate, iters, params):
        self.learning_rate = learning_rate
        self.iters = iters
        self.params = params

    def erro(self, params):
        a, b, x, y = params
        yhat = a.T @ x + b
        mse = np_.mean((yhat - y)**2)
        return mse

    def melhorar_modelo(self):
        params = self.params
        for i in range(self.iters):
            a, b, x, y = params
            grads = grad(self.erro)(params)
            a -= self.learning_rate * grads[0]
            b -= self.learning_rate * grads[1]
            params = [a, b, x, y]
        return params

    @staticmethod
    def predict(params, xtest):
        a, b, x, y = params
        ypred = a.T @ xtest + b
        return ypred

    @staticmethod
    def accuracy(ypred, ytest):
        return np.mean(np.sign(ypred) == np.sign(ytest))

    def main(self):
        params = self.melhorar_modelo()
        return params
