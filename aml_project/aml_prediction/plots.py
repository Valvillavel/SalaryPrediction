"""Funciones de visualización reutilizables (equivalentes a tus celdas de matplotlib/seaborn)."""

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def plot_class_distribution(y, title="Distribución de la variable objetivo"):
    """Grafica el conteo de clases (equivalente a tu Figura de balance de clases)."""
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(x=y, ax=ax)
    ax.set_title(title)
    return fig


def plot_confusion(y_true, y_pred, labels=("<=50K", ">50K")):
    """Grafica la matriz de confusión."""
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    return fig