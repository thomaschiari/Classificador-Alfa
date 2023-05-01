import numpy as np
import autograd.numpy as np_
from autograd import grad


class ClassificadorAlfa():

    def __init__(self, learning_rate, iters, params):
        self.learning_rate = learning_rate
        self.iters = iters
        self.params = params

    @staticmethod
    def erro(params):
        a, b, x, y = params
        yhat = a.T@x + b
        mse = np_.mean((yhat - y)**2)
        return mse

    def melhorar_modelo(self):
        g = grad(ClassificadorAlfa.erro)
        a, b, x, y = self.params
        for _ in range(self.iters):
            g_ = g((a, b, x, y))
            a -= self.learning_rate * g_[0]
            b -= self.learning_rate * g_[1]
        return a, b

    def treinar(self):
        a, b = self.melhorar_modelo()
        return a, b

    @staticmethod
    def acuracia(ytest, ypred):
        return np.mean(np.sign(ytest) == np.sign(ypred))
