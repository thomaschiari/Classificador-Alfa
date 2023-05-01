import numpy as np
import autograd.numpy as np_
from autograd import grad


class ClassificadorAlfa():

    def __init__(self, learning_rate, iters, params):
        """
        Inicializa o objeto ClassificadorAlfa com os parâmetros necessários para o treinamento.
        
        Args:
        learning_rate (float): taxa de aprendizagem que controla o tamanho do passo que o algoritmo de otimização dá em cada iteração.
        iters (int): número de iterações que o algoritmo de otimização executará.
        params (list): lista contendo os parâmetros iniciais do modelo (a, b, x, y), onde:
                        a (array_like): array com os coeficientes do modelo.
                        b (float): termo de bias do modelo.
                        x (array_like): array com as features do conjunto de dados.
                        y (array_like): array com os valores de saída do conjunto de dados.
        """
        self.learning_rate = learning_rate
        self.iters = iters
        self.params = params

    @staticmethod
    def erro(params):
        """
        Calcula o erro quadrático médio (MSE) do modelo com os parâmetros passados como argumento.
        
        Args:
        params (tuple): tupla contendo os parâmetros do modelo (a, b, x, y), onde:
                        a (array_like): array com os coeficientes do modelo.
                        b (float): termo de bias do modelo.
                        x (array_like): array com as features do conjunto de dados.
                        y (array_like): array com os valores de saída do conjunto de dados.
                        
        Returns:
        float: valor do MSE calculado.
        """
        a, b, x, y = params
        yhat = a.T@x + b #calcula a predição do modelo para as features x
        mse = np_.mean((yhat - y)**2) #calcula o MSE entre a predição do modelo e os valores reais y
        return mse

    def melhorar_modelo(self):
        """
        Otimiza os parâmetros do modelo para reduzir o erro calculado pela função erro.
        
        Returns:
        tuple: tupla contendo os valores otimizados dos parâmetros a e b.
        """
        g = grad(ClassificadorAlfa.erro) #calcula a função gradiente da função erro
        a, b, x, y = self.params
        for _ in range(self.iters):
            g_ = g((a, b, x, y)) #calcula o gradiente da função erro para os parâmetros atuais
            a -= self.learning_rate * g_[0] #atualiza o valor de a usando o gradiente
            b -= self.learning_rate * g_[1] #atualiza o valor de b usando o gradiente
        return a, b

    def treinar(self):
        """
        Treina o modelo para encontrar os melhores valores dos parâmetros.
        
        Returns:
        tuple: tupla contendo os valores otimizados dos parâmetros a e b.
        """
        a, b = self.melhorar_modelo() #otimiza os parâmetros do modelo
        return a, b
        
    @staticmethod
    def acuracia(ytest, ypred):
        """
        Calcula a acurácia do modelo comparando as classes preditas com as classes reais.
        
        Args:
        ytest (array_like): array com as classes reais do conjunto de teste.
        ypred (array_like): array com as classes preditas pelo modelo.
        
        Returns:
        float: valor da acurácia calculada.
        """
        return np.mean(np.sign(ytest) == np.sign(ypred)) #compara as classes reais e as preditas e calcula a média dos acertos. Retorna a acurácia.

