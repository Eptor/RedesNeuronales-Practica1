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
from sklearn.decomposition import PCA

PATH = "data/wdbc.csv"

features_names = {
    20: "symmetry2",
    7: "compactness1",
    22: "radius3",
    27: "compactness3",
    18: "concavity2",
    11: "symmetry1",
    9: "concave_points1",
    14: "perimeter2",
}

def calculate_metrics(y_true, y_pred):
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
    df = pd.read_csv(PATH, header=None)
    y = np.array(df.iloc[:, 1].replace({"M": 1, "B": 0}), dtype=int)
    X_raw = np.array(df.iloc[:, 2:], dtype=float)

    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)

    # Relevancia
    f_scores = fisher_score(X, y)
    best_indices = np.argsort(f_scores)[-4:]  # Las 4 mejores
    worst_indices = np.argsort(f_scores)[:4]  # Las 4 peores

    print("Las 4 características más relevantes según el criterio de Fisher:")
    for idx in best_indices:
        print(f"  - {features_names.get(idx, f'Feature {idx}')} (Fisher Score: {f_scores[idx]:.4f})")

    print("\nLas 4 características menos relevantes según el criterio de Fisher:")
    for idx in worst_indices:
        print(f"  - {features_names.get(idx, f'Feature {idx}')} (Fisher Score: {f_scores[idx]:.4f})")

    # Visualización 2D (dispersión) usando PCA a 2 componentes
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    fig_pca, ax_pca = plt.subplots(figsize=(8, 6))
    palette = {0: '#1f77b4', 1: '#d62728'}
    labels_map = {0: 'Benigno', 1: 'Maligno'}
    for label in np.unique(y):
        mask = y == label
        ax_pca.scatter(
            X_pca[mask, 0],
            X_pca[mask, 1],
            c=palette[label],
            label=labels_map[label],
            edgecolor='k',
            s=40,
            alpha=0.8,
        )

    ax_pca.set_xlabel('PC 1')
    ax_pca.set_ylabel('PC 2')
    ax_pca.set_title('Proyección PCA 2D - dispersión por clase')
    ax_pca.legend()
    ax_pca.grid(alpha=0.3)

    fig_pca.show()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {
        "Regresión Logística": LogisticRegression(),
        "Random Forest": RandomForestClassifier(n_estimators=100),
        "SVM (Linear)": SVC(kernel="linear"),
    }

    fig_cm, axes_cm = plt.subplots(1, 3, figsize=(18, 5))
    fig_cm.suptitle("Comparación de Matrices de Confusión", fontsize=16)

    metrics_data = {}

    for i, (name, model) in enumerate(models.items()):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        cm = confusion_matrix(y_test, y_pred)

        # Normalizar por filas para obtener porcentajes por clase real
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_percent = np.divide(
            cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0
        ) * 100.0

        # Crear anotaciones con porcentaje y recuento: "xx.x%\n(count)"
        annot = np.empty(cm_percent.shape, dtype=object)
        for r in range(cm_percent.shape[0]):
            for c in range(cm_percent.shape[1]):
                annot[r, c] = f"{cm_percent[r, c]:.1f}%\n({cm[r, c]})"

        sns.heatmap(
            cm_percent,
            annot=annot,
            fmt="",
            cmap="Greens",
            ax=axes_cm[i],
            vmin=0,
            vmax=100,
            cbar_kws={"label": "Porcentaje (%)"},
        )
        axes_cm[i].set_title(f"{name}\nAcc: {accuracy_score(y_test, y_pred):.3f}")
        axes_cm[i].set_xlabel("Predicho")
        axes_cm[i].set_ylabel("Real")

        metrics = calculate_metrics(y_test, y_pred)
        metrics_data[name] = metrics
        
        print(f"\n{name}:")
        print(f"  Exactitud: {metrics['Accuracy']:.4f}")
        print(f"  Precisión: {metrics['Precision']:.4f}")
        print(f"  Sensibilidad (Recall): {metrics['Sensitivity']:.4f}")
        print(f"  Especificidad: {metrics['Specificity']:.4f}")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
    
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
