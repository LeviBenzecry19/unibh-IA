import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np

class Perceptron:
    def __init__(self, taxa_aprendizado=0.01, n_iter=100, random_state=1):
        self.taxa_aprendizado = taxa_aprendizado  # Eta (η)
        self.n_iter = n_iter
        self.pesos = None       # Vetor de pesos
        self.bias = None        # Viés
        self.registro_erros = []  # Para monitorar o aprendizado

    def fit(self, X, y):
        n_amostras, n_features = X.shape
        # Inicializa pesos e bias (pode ser aleatório ou zero)
        self.pesos = np.zeros(n_features)
        self.bias = 0
        self.registro_erros = []

        for _ in range(self.n_iter):
            erros = 0
            for xi, target in zip(X, y):
                # Previsão: função degrau (Heaviside)
                pred = self.ativacao(np.dot(xi, self.pesos) + self.bias)

                # Atualização dos pesos e bias (Aprendizado Supervisionado)
                erro = target - pred
                if erro != 0:
                    self.pesos += self.taxa_aprendizado * erro * xi
                    self.bias  += self.taxa_aprendizado * erro
                    erros += 1

            self.registro_erros.append(erros)
        return self

    def ativacao(self, z):
        # Função de ativação degrau (Heaviside)
        return np.where(z >= 0, 1, 0)

    def predict(self, X):
        # Aplica a função de ativação nas somas ponderadas
        return self.ativacao(np.dot(X, self.pesos) + self.bias)


# --- Exemplo de Uso (Porta Lógica AND) ---
# Dados de entrada (X) e saídas esperadas (y) para a porta AND
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 0, 0, 1])

# Cria e treina o perceptron
ppn = Perceptron(taxa_aprendizado=0.1, n_iter=10)
ppn.fit(X, y)

print("Pesos após treinamento:", ppn.pesos)
print("Bias após treinamento:", ppn.bias)
print("Previsões:", ppn.predict(X))


# Cria e treina o perceptron - teste com porta OR
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 1])
ppn = Perceptron(taxa_aprendizado=0.1, n_iter=10)
ppn.fit(X, y)
print("Treinamento da porta OR")
print("Pesos após treinamento:", ppn.pesos)
print("Bias após treinamento:", ppn.bias)
print("Previsões:", ppn.predict(X))


# Cria e treina o perceptron - teste com porta OR e 3 entradas
X = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]])
y = np.array([0, 1, 1, 1, 1, 1, 1, 1])
ppn = Perceptron(taxa_aprendizado=0.1, n_iter=10)
ppn.fit(X, y)
print("Treinamento da porta OR de 3 entradas")
print("Pesos após treinamento:", ppn.pesos)
print("Bias após treinamento:", ppn.bias)
print("Previsões:", ppn.predict(X))
