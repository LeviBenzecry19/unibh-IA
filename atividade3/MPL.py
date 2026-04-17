import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.metrics import classification_report, roc_curve, auc, RocCurveDisplay

# Carregar o conjunto de dados
data = load_breast_cancer()
X = data.data
y = data.target  

print(data.DESCR)
   
# Dividir o conjunto de dados em treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Normalizar os dados
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Criar e treinar o modelo MLP
mlp = MLPClassifier(hidden_layer_sizes=(100,50), max_iter=500, random_state=42, activation='relu', solver='adam', verbose=False)

# Treinar o modelo
mlp.fit(X_train_scaled, y_train)

# Loss Curve
plt.figure(figsize=(10, 5))
plt.plot(mlp.loss_curve_)
plt.title('Loss Curve')
plt.xlabel('Épocas')
plt.ylabel('Loss')
plt.show()
plt.grid(True)

# Avaliar o modelo
y_pred = mlp.predict_proba(X_test_scaled)[:,1]  # Probabilidades para a classe positiva
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)

# Exibir a curva ROC
plt.figure(figsize=(10, 5))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

#relatório de classificação
y_pred = mlp.predict(X_test_scaled)
print("Relatório de Classificação:")
print(classification_report(y_test, y_pred, target_names=data.target_names))