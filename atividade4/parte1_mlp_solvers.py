import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# =============================================================================
# Atividade 4 - Parte 1: MLP com diferentes Solvers
# Disciplina: Inteligência Artificial - UniBH 2026
# Professor: Alexandre "Montanha" de Oliveira
# =============================================================================
# Objetivo: Rodar a MLP com cada um dos algoritmos e comparar a melhor AUC
# Solvers: 'lbfgs', 'sgd', 'adam'
# =============================================================================

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_curve, auc, classification_report, RocCurveDisplay
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# 1. Carregar o dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Conheça sua base de dados
print(data.DESCR)

# 2. Dividir em treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 3. Normalização (Essencial para redes neurais)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =============================================================================
# Comparação dos 3 Solvers
# =============================================================================
solvers = ['lbfgs', 'sgd', 'adam']
resultados = {}

print("\n" + "=" * 60)
print("COMPARAÇÃO DE SOLVERS - MLP")
print("=" * 60)

fig_roc, ax_roc = plt.subplots(figsize=(10, 7))
ax_roc.plot([0, 1], [0, 1], color='navy', lw=1.5, linestyle='--', label='Aleatório (AUC = 0.50)')

for solver in solvers:
    print(f"\n{'=' * 60}")
    print(f"Solver: {solver.upper()}")
    print(f"{'=' * 60}")

    # 4. Criar e treinar o MLP
    mlp = MLPClassifier(
        hidden_layer_sizes=(100, 50),
        max_iter=500,
        activation='relu',
        solver=solver,
        random_state=42,
        verbose=False
    )

    mlp.fit(X_train_scaled, y_train)
    print(f"Épocas necessárias: {mlp.n_iter_}")

    # 5. Visualização do Erro (Loss Curve)
    # lbfgs não expõe loss_curve_ (otimizador em batch, não iterativo por época)
    if hasattr(mlp, 'loss_curve_'):
        plt.figure(figsize=(10, 5))
        plt.plot(mlp.loss_curve_, lw=2)
        plt.title(f"Curva de Erro durante Treinamento (Loss Curve) - Solver: {solver}")
        plt.xlabel("Épocas")
        plt.ylabel("Erro (Loss)")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    else:
        print(f"  (Solver '{solver}' não gera Loss Curve por época)")

    # 6. Avaliação e Gráfico AUC-ROC
    y_pred_proba = mlp.predict_proba(X_test_scaled)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    # Plotar ROC individual
    plt.figure(figsize=(10, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Taxa de Falsos Positivos')
    plt.ylabel('Taxa de Verdadeiros Positivos')
    plt.title(f'Curva ROC - MLP Breast Cancer (Solver: {solver})')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Adicionar ao gráfico comparativo
    ax_roc.plot(fpr, tpr, lw=2, label=f'{solver.upper()} (AUC = {roc_auc:.4f})')

    # 7. Relatório de Classificação
    y_pred = mlp.predict(X_test_scaled)
    print("\nRelatório de Classificação:\n")
    print(classification_report(y_test, y_pred, target_names=data.target_names))

    # Matriz de confusão
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=data.target_names)
    fig_cm, ax_cm = plt.subplots(figsize=(7, 5))
    disp.plot(ax=ax_cm, cmap='Blues')
    ax_cm.set_title(f'Matriz de Confusão - Solver: {solver}')
    plt.tight_layout()
    plt.show()

    # Salvar resultados
    resultados[solver] = {
        'AUC': roc_auc,
        'épocas': mlp.n_iter_,
        'loss_final': mlp.loss_curve_[-1] if hasattr(mlp, 'loss_curve_') else mlp.loss_
    }

# Gráfico ROC comparativo (todos os solvers juntos)
ax_roc.set_xlim([0.0, 1.0])
ax_roc.set_ylim([0.0, 1.05])
ax_roc.set_xlabel('Taxa de Falsos Positivos', fontsize=13)
ax_roc.set_ylabel('Taxa de Verdadeiros Positivos', fontsize=13)
ax_roc.set_title('Comparação de Curvas ROC - MLP por Solver', fontsize=14, fontweight='bold')
ax_roc.legend(loc='lower right', fontsize=12)
ax_roc.grid(True, alpha=0.3)
fig_roc.tight_layout()
plt.show()

# =============================================================================
# Tabela Comparativa Final
# =============================================================================
print("\n" + "=" * 60)
print("TABELA COMPARATIVA - RESULTADOS POR SOLVER")
print("=" * 60)
print(f"\n{'Solver':<10} {'AUC':<10} {'Épocas':<10} {'Loss Final':<12}")
print("-" * 45)
for solver, res in resultados.items():
    print(f"{solver:<10} {res['AUC']:.4f}    {res['épocas']:<10} {res['loss_final']:.4f}")

melhor = max(resultados, key=lambda s: resultados[s]['AUC'])
print(f"\n🏆 Melhor solver: {melhor.upper()} (AUC = {resultados[melhor]['AUC']:.4f})")

print("\n💡 Análise dos Solvers:")
print("-" * 60)
print("• lbfgs : Quasi-Newton, bom para datasets pequenos/médios")
print("• sgd   : Gradiente estocástico, simples mas pode ser lento")
print("• adam  : Adaptativo, geralmente o melhor para redes neurais")
