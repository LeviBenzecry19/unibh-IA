import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Importar bibliotecas necessárias
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# 1. Carregar o conjunto de dados
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

# 2. Dividir os dados em conjuntos de treino e teste
# 75% treino, 25% teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# 3. Treinar o modelo de Árvore de Decisão
# Usamos o critério 'entropy' (entropia) para medir a qualidade da divisão
# Outra opção comum é 'gini' (índice Gini)
classifier = DecisionTreeClassifier(criterion='entropy', random_state=0)
classifier.fit(X_train, y_train)

# 4. Fazer previsões no conjunto de teste
y_pred = classifier.predict(X_test)

# 5. Avaliar o desempenho do modelo
accuracy = accuracy_score(y_test, y_pred)
print(f"Acurácia do modelo: {accuracy:.2f}")
print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred, target_names=target_names))

# 6. Visualizar a Árvore de Decisão
plt.figure(figsize=(12, 8))
plot_tree(classifier,
          feature_names=feature_names,
          class_names=target_names,
          filled=True,
          rounded=True,
          proportion=True,
          fontsize=10)
plt.title("Árvore de Decisão do Conjunto de Dados Iris")
plt.show()
