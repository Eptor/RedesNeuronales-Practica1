import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

PATH = "data/wdbc.csv"


def calculate_metrics(y_true, y_pred):
    """Calcula métricas de desempeño: accuracy, precision, sensitivity, specificity."""
    # Accuracy
    acc = accuracy_score(y_true, y_pred)
    
    # Precision
    prec = precision_score(y_true, y_pred)
    
    # Sensitivity (Recall)
    sens = recall_score(y_true, y_pred)
    
    # Specificity: TN / (TN + FP)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    return {'Accuracy': acc, 'Precision': prec, 'Sensitivity': sens, 'Specificity': spec}


def fisher_score(X, y):
    """Calcula el Fisher Score para cada columna."""
    labels = np.unique(y)
    scores = []
    for i in range(X.shape[1]):
        feature = X[:, i]
        m = np.mean(feature)
        numerator = 0
        denominator = 0
        for label in labels:
            X_c = feature[y == label]
            m_c = np.mean(X_c)
            n_c = len(X_c)
            v_c = np.var(X_c)
            numerator += n_c * (m_c - m) ** 2
            denominator += n_c * v_c
        scores.append(numerator / (denominator + 1e-6))
    return np.array(scores)


def main():
    # 1. Cargar y Preprocesar
    df = pd.read_csv(PATH, header=None)
    y = np.array(df.iloc[:, 1].replace({"M": 1, "B": 0}), dtype=int)
    X_raw = np.array(df.iloc[:, 2:], dtype=float)

    # NORMALIZACIÓN
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)

    # 2. Análisis de Relevancia (Fisher)
    f_scores = fisher_score(X, y)
    best_indices = np.argsort(f_scores)[-4:]  # Las 4 mejores

    # GRAFICAR CARACTERÍSTICAS (Subplots 2x2)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(
        "Distribución de las 4 Características más Relevantes (Fisher Score)",
        fontsize=16,
    )

    for i, idx in enumerate(best_indices):
        ax = axes[i // 2, i % 2]
        sns.histplot(x=X[:, idx], hue=y, kde=True, ax=ax, palette="RdBu")
        ax.set_title(f"Característica {idx} - Score: {f_scores[idx]:.2f}")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    # 3. Entrenamiento de Modelos
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {
        "Regresión Logística": LogisticRegression(),
        "Random Forest": RandomForestClassifier(n_estimators=100),
        "SVM (Linear)": SVC(kernel="linear"),
    }

    # GRAFICAR MATRICES DE CONFUSIÓN
    fig_cm, axes_cm = plt.subplots(1, 3, figsize=(18, 5))
    fig_cm.suptitle("Comparación de Matrices de Confusión", fontsize=16)

    # Almacenar métricas para gráfico comparativo
    metrics_data = {}

    for i, (name, model) in enumerate(models.items()):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", ax=axes_cm[i])
        axes_cm[i].set_title(f"{name}\nAcc: {accuracy_score(y_test, y_pred):.3f}")
        axes_cm[i].set_xlabel("Predicho")
        axes_cm[i].set_ylabel("Real")
        
        # Calcular métricas
        metrics = calculate_metrics(y_test, y_pred)
        metrics_data[name] = metrics
        
        # Imprimir métricas
        print(f"\n{name}:")
        print(f"  Exactitud: {metrics['Accuracy']:.4f}")
        print(f"  Precisión: {metrics['Precision']:.4f}")
        print(f"  Sensibilidad (Recall): {metrics['Sensitivity']:.4f}")
        print(f"  Especificidad: {metrics['Specificity']:.4f}")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
    
    # 4. Gráfico de barras comparativo
    metrics_names = list(metrics_data.values())[0].keys()
    x = np.arange(len(metrics_names))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for i, (name, metrics) in enumerate(metrics_data.items()):
        values = [metrics[metric] for metric in metrics_names]
        ax.bar(x + i * width, values, width, label=name)
    
    ax.set_xlabel('Métricas', fontsize=12)
    ax.set_ylabel('Valor', fontsize=12)
    ax.set_title('Comparación de Desempeño de Algoritmos', fontsize=14)
    ax.set_xticks(x + width)
    ax.set_xticklabels(metrics_names)
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', alpha=0.3)
    
    # Agregar valores en las barras
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', padding=3)
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
