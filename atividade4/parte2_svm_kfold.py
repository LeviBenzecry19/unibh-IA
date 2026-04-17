import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# =============================================================================
# Atividade 4 - Parte 2: SVM com K-Fold Cross Validation
# Disciplina: Inteligência Artificial - UniBH 2026
# Professor: Alexandre "Montanha" de Oliveira
# =============================================================================
# Objetivo:
#   - Rodar SVM com diferentes kernels
#   - Variar K no K-Fold
#   - Verificar se interfere no resultado
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import (
    train_test_split, KFold, cross_val_score, StratifiedKFold
)
from sklearn.metrics import (
    roc_curve, auc, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.pipeline import Pipeline

# 1. Carregar o dataset
data = load_breast_cancer()
X = data.data
y = data.target

print("=" * 60)
print("ATIVIDADE 4 - PARTE 2: SVM COM K-FOLD CROSS VALIDATION")
print("=" * 60)
print(f"\nDataset: Breast Cancer Wisconsin")
print(f"Amostras: {X.shape[0]} | Features: {X.shape[1]}")
print(f"Classe 0 (maligno): {np.sum(y == 0)} | Classe 1 (benigno): {np.sum(y == 1)}")

# 2. Dividir em treino e teste (holdout final)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Normalização
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =============================================================================
# PARTE A: Comparação de Kernels SVM
# =============================================================================
print("\n" + "=" * 60)
print("PARTE A: COMPARANDO KERNELS DO SVM")
print("=" * 60)
print("\nKernels disponíveis: linear, rbf, poly, sigmoid")
print("Pergunta: Variar o kernel interfere no resultado?\n")

kernels = ['linear', 'rbf', 'poly', 'sigmoid']
resultados_kernel = {}

fig_roc, ax_roc = plt.subplots(figsize=(10, 7))
ax_roc.plot([0, 1], [0, 1], color='navy', lw=1.5,
            linestyle='--', label='Aleatório (AUC = 0.50)')

for kernel in kernels:
    print(f"Treinando SVM com kernel='{kernel}'...")

    svm = SVC(kernel=kernel, probability=True, random_state=42)
    svm.fit(X_train_scaled, y_train)

    # Avaliação
    y_pred = svm.predict(X_test_scaled)
    y_proba = svm.predict_proba(X_test_scaled)[:, 1]

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)

    from sklearn.metrics import accuracy_score
    acc = accuracy_score(y_test, y_pred) * 100

    resultados_kernel[kernel] = {'AUC': roc_auc, 'Acurácia': acc}
    ax_roc.plot(fpr, tpr, lw=2, label=f'{kernel} (AUC = {roc_auc:.4f})')

    print(f"  → AUC: {roc_auc:.4f} | Acurácia: {acc:.2f}%")

    # Relatório
    print(f"\n  Relatório de Classificação (kernel={kernel}):")
    print(classification_report(y_test, y_pred,
                                 target_names=data.target_names, zero_division=0))

    # Matriz de confusão
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                   display_labels=data.target_names)
    fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax_cm, cmap='Blues')
    ax_cm.set_title(f'Matriz de Confusão - SVM kernel={kernel}')
    plt.tight_layout()
    plt.show()

# Gráfico ROC comparativo de kernels
ax_roc.set_xlim([0.0, 1.0])
ax_roc.set_ylim([0.0, 1.05])
ax_roc.set_xlabel('Taxa de Falsos Positivos', fontsize=13)
ax_roc.set_ylabel('Taxa de Verdadeiros Positivos', fontsize=13)
ax_roc.set_title('Comparação de Kernels SVM - Curvas ROC', fontsize=14, fontweight='bold')
ax_roc.legend(loc='lower right', fontsize=12)
ax_roc.grid(True, alpha=0.3)
fig_roc.tight_layout()
plt.show()

# Tabela kernels
print("\n" + "=" * 60)
print("TABELA: COMPARAÇÃO DE KERNELS")
print("=" * 60)
print(f"\n{'Kernel':<12} {'AUC':<10} {'Acurácia':<12}")
print("-" * 38)
for k, res in resultados_kernel.items():
    print(f"{k:<12} {res['AUC']:.4f}    {res['Acurácia']:.2f}%")

melhor_kernel = max(resultados_kernel, key=lambda k: resultados_kernel[k]['AUC'])
print(f"\n🏆 Melhor kernel: {melhor_kernel} (AUC = {resultados_kernel[melhor_kernel]['AUC']:.4f})")


# =============================================================================
# PARTE B: Variando K no K-Fold Cross Validation
# =============================================================================
print("\n" + "=" * 60)
print("PARTE B: VARIANDO K NO K-FOLD CROSS VALIDATION")
print("=" * 60)
print("\nPergunta: Variar o K interfere no resultado?\n")

# Usar pipeline para garantir normalização dentro de cada fold
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC(kernel='rbf', probability=True, random_state=42))
])

valores_k = [2, 3, 5, 10, 15, 20]
resultados_kfold = {}

for k in valores_k:
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    scores = cross_val_score(pipeline, X, y, cv=skf, scoring='roc_auc')

    resultados_kfold[k] = {
        'media': scores.mean(),
        'desvio': scores.std(),
        'scores': scores
    }
    print(f"K={k:2d} | AUC médio: {scores.mean():.4f} ± {scores.std():.4f}")

# Gráfico: AUC médio por K
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Gráfico 1: AUC médio com barra de erro
ks = list(resultados_kfold.keys())
medias = [resultados_kfold[k]['media'] for k in ks]
desvios = [resultados_kfold[k]['desvio'] for k in ks]

axes[0].errorbar(ks, medias, yerr=desvios, marker='o', linewidth=2,
                 markersize=8, capsize=5, color='blue', ecolor='red')
axes[0].set_xlabel('Valor de K', fontsize=13)
axes[0].set_ylabel('AUC Médio (Cross-Validation)', fontsize=13)
axes[0].set_title('Impacto do K no K-Fold Cross Validation\n(SVM kernel=rbf)', fontsize=13, fontweight='bold')
axes[0].grid(True, alpha=0.3)
axes[0].set_xticks(ks)

# Gráfico 2: Boxplot dos scores por K
box_data = [resultados_kfold[k]['scores'] for k in ks]
axes[1].boxplot(box_data, labels=[str(k) for k in ks])
axes[1].set_xlabel('Valor de K', fontsize=13)
axes[1].set_ylabel('AUC por Fold', fontsize=13)
axes[1].set_title('Distribuição do AUC por Fold\n(SVM kernel=rbf)', fontsize=13, fontweight='bold')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Tabela K-Fold
print("\n" + "=" * 60)
print("TABELA: IMPACTO DO K NO K-FOLD")
print("=" * 60)
print(f"\n{'K':<6} {'AUC Médio':<12} {'Desvio Padrão':<15} {'Interpretação'}")
print("-" * 60)
for k in ks:
    res = resultados_kfold[k]
    interp = "Alta variância" if res['desvio'] > 0.03 else "Estável"
    print(f"{k:<6} {res['media']:.4f}       {res['desvio']:.4f}          {interp}")


# =============================================================================
# PARTE C: Matriz Final - Melhor Kernel × Melhor K
# =============================================================================
print("\n" + "=" * 60)
print("PARTE C: MELHOR CONFIGURAÇÃO FINAL")
print("=" * 60)

melhor_k = min(resultados_kfold, key=lambda k: resultados_kfold[k]['desvio'])
print(f"\nMelhor kernel: {melhor_kernel}")
print(f"K recomendado: {melhor_k} (menor desvio padrão)")

# Treinar modelo final com melhor configuração
svm_final = SVC(kernel=melhor_kernel, probability=True, random_state=42)
svm_final.fit(X_train_scaled, y_train)

y_final_pred = svm_final.predict(X_test_scaled)
y_final_proba = svm_final.predict_proba(X_test_scaled)[:, 1]
fpr_f, tpr_f, _ = roc_curve(y_test, y_final_proba)
auc_final = auc(fpr_f, tpr_f)

print(f"\n📊 Resultado do Modelo Final (kernel={melhor_kernel}):")
print(classification_report(y_test, y_final_pred, target_names=data.target_names))
print(f"AUC Final: {auc_final:.4f}")

# Curva ROC final
plt.figure(figsize=(9, 6))
plt.plot(fpr_f, tpr_f, color='darkorange', lw=2,
         label=f'SVM {melhor_kernel} (AUC = {auc_final:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=1.5, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Taxa de Falsos Positivos', fontsize=13)
plt.ylabel('Taxa de Verdadeiros Positivos', fontsize=13)
plt.title(f'Curva ROC Final - SVM ({melhor_kernel})', fontsize=14, fontweight='bold')
plt.legend(loc='lower right', fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# =============================================================================
# Conclusão
# =============================================================================
print("\n" + "=" * 60)
print("CONCLUSÃO DA ATIVIDADE 4 - PARTE 2")
print("=" * 60)
print("""
📌 Variar o kernel interfere no resultado?
   SIM. Kernels capturam diferentes geometrias de separação:
   • linear : hiperplano linear (bom para dados linearmente separáveis)
   • rbf    : radial basis function (mais flexível, geralmente melhor)
   • poly   : fronteira polinomial (complexo, pode overfittar)
   • sigmoid: similar a redes neurais

📌 Variar K no K-Fold interfere no resultado?
   SIM, mas sutilmente. Tradeoff Bias-Variância:
   • K pequeno (2-3)  → Maior bias, menor variância
   • K grande (10-20) → Menor bias, maior variância
   • K = 5 ou K = 10 → Equilíbrio recomendado (empírico)
   • K = n (LOOCV)   → Mínimo bias, máxima variância computacional

✅ Recomendação prática: SVM com kernel RBF + K-Fold com K=10
""")
