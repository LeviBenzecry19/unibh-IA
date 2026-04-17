import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler

# 1. Gerar/Carregar dados (Simulando "Mall_Customers.csv")
np.random.seed(42)
# Criando 3 grupos principais com algum ruído entre eles
data = np.vstack([
    np.random.normal(loc=[20, 80], scale=5,  size=(30, 2)),  # Jovens Gastadores
    np.random.normal(loc=[40, 50], scale=10, size=(40, 2)),  # Gastadores Moderados (Área de overlap)
    np.random.normal(loc=[50, 20], scale=5,  size=(30, 2))   # Mais velhos, Gastam pouco
])

# Normalizar os dados (Crucial para algoritmos baseados em distância)
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)
X = data_scaled.T  # scikit-fuzzy precisa de (features, samples)

# 2. Configurar o algoritmo Fuzzy C-Means
n_clusters = 3      # Definindo o número de segmentos (c)
m = 2.0             # Parâmetro de "fuzziness" (geralmente 2)
error = 0.005       # Critério de parada
maxiter = 1000      # Máximo de iterações

# 3. Executar o FCM
cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
    X, c=n_clusters, m=m, error=error, maxiter=maxiter, init=None
)

# 4. Analisar os resultados
# Obter o cluster com maior pertinência para cada ponto
cluster_membership = np.argmax(u, axis=0)

# 5. Visualizar
fig, ax = plt.subplots(figsize=(10, 6))
colors = ['r', 'g', 'b']

for j in range(n_clusters):
    # Selecionar pontos pertencentes ao cluster j (hard assignment para visualização)
    points = data[cluster_membership == j]
    ax.scatter(points[:, 0], points[:, 1], c=colors[j], label=f'Segmento {j+1}', alpha=0.6)

# Plotar os centroides
# Converter centros de volta para escala original para visualização
centers_orig = scaler.inverse_transform(cntr)
ax.scatter(centers_orig[:, 0], centers_orig[:, 1], marker='x', s=200, c='black', linewidths=3)

ax.set_title(f'Segmentação de Clientes Fuzzy C-Means (n_clusters={n_clusters})')
ax.set_xlabel('Idade')
ax.set_ylabel('Pontuação de Gastos (1-100)')
ax.legend()
plt.show()

# Exemplo de pertinência (Fuzzy): Mostrar a qual cluster os primeiros 5 pontos pertencem
print("Graus de Pertinência (primeiros 5 clientes) [Cluster1, Cluster2, Cluster3]:")
print(u[:, :5].T)
