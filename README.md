# Classificador-Alfa
Utilização de classificadores para identificar fatores de risco para um AVC.
Os arquivos estão separados em `ClassificadorAVC.ipynb`, que possui o classificador completo, incluindo a análise exploratória realizada pelo grupo; `ClassificadorAlfa.py`, que define as funções de classificação a serem utilizadas pelo grupo, 
e `demo.ipynb`, que contém apenas um exemplo de utilização do classificador, sem a análise exploratória.

## Autores
- Marcelo Rabello Barranco
- Thomas Chiari Ciocchetti de Souza

## Utilização
1. Verifique se possui Python 3.9 ou superior instalado em sua máquina.
2. Clone o repositório em sua máquina utilizando o comando `git clone https://github.com/thomaschiari/Classificador-Alfa.git`
3. Instale as dependências necessárias utilizando o comando `pip install -r requirements.txt`
4. Execute o arquivo `demo.ipynb` para ver um exemplo de utilização do classificador.
5. Caso deseje, importe o arquivo `ClassificadorAlfa.py` para utilizar as funções de classificação em seu próprio código.

## Resumo

### Metodologia

Para realizar a análise de fatores de risco para um AVC, o grupo realizou uma análise comparativa entre três classificadores: um classificador linear, realizado pelo grupo e presente no arquivo [`ClassificadorAlfa`](ClassificadorAlfa.py), que utiliza o algoritmo de gradiente descentente para encontrar os melhores parâmetros para o modelo; uma árvore de decisão, para a qual utilizamos a biblioteca scikit-learn, através da metodologia de entropia; e um classificador que simplesmente prevê a observação mais frequente do dataset (aqui considerada a hipótese nula). Realizamos comparações para as acurácias dos três modelos, além de obtermos as características mais relevantes para prever um AVC.

### Resultados

O grupo pôde observar que, para as bases de dados completas, não houve diferenças significativas entre os classificadores elaborados e a hipótese nula. Isso se deu pois na base completa, cerca de 95% dos dados presentes são de pessoas que não tiveram AVC, então o classificador nulo sempre terá uma acurácia de 95%, considerada alta. 

Ao realizarmos o teste com as bases balanceadas, com 50% de dados de pessoas que tiveram AVCs e 50% que não tiveram, obtivemos diferenças significativas entre os classificadores. A hipótese nula se manteve em 50%, enquanto os classificadores obtiveram resultados superiores, e conseguiram identificar características mais relevantes que podem ser considerados fatores de risco para um AVC. 

O grupo obteve, entre as características mais importantes, a hipertensão, presença de doenças cardíacas, gênero, tabagismo e local de trabalho dos indivíduos analisados. Tais observações, com exceção do local de trabalho, são corroboradas por O'Donnel, J. apud Boehme, A. no artigo ["Stroke Risk Factors, Genetics, and Prevention"](https://www.ahajournals.org/doi/full/10.1161/CIRCRESAHA.116.308398). O local de trabalho pode ter uma importância significativa, apesar de não ser corroborado pelo artigo, pois pode ter alguma correlação com outros fatores de risco de um AVC não contemplados nos dados, como dieta, atividade física ou consumo de bebidas alcoólicas dos indivíduos. Tais análises estão disponíveis no arquivo [`demo.ipynb`](demo.ipynb).

Em suma, foi possível descobrir e embasar alguns dos principais fatores de risco para um AVC. No geral, os classificadores podem não ter tido uma diferença muito significativa de acurácia frente à hipótese nula, mas forneceram uma boa base de quais características são importantes para a prevenção de um AVC. A principal característica citada no artigo, hipertensão, foi definida com uma das maiores importâncias nos modelos elaborados. Demais fatores, como presença de doença cardíaca e tabagismo, também se mostraram importantes nas duas análises. 

## Referências
- Boehme, A., Esenwa, C., Elkind, M. Stroke Risk Factors, Genetics, and Prevention. Circulation Research, Feb. 2017. Available at [Circulation Research](https://www.ahajournals.org/doi/full/10.1161/CIRCRESAHA.116.308398)
- O'Donnel, J. et al. Risk factors for ischaemic and intracerebral haemorrhagic stroke in 22 countries (the INTERSTROKE study): a case-control study. The Lancet, Jun. 2010. Available at <a href="https://doi.org/10.1016/S0140-6736(10)60834-3">The Lancet</a>


