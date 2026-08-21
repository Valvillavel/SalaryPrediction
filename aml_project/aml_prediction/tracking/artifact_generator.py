"""Genera gráficos de diagnóstico para guardarlos como artefactos en MLflow."""

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, RocCurveDisplay


class ArtifactGenerator:
    """Genera gráficos de diagnóstico (matriz de confusión, curva ROC)."""

    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, output_path="confusion_matrix.png"):
        """Guarda la matriz de confusión como imagen.

        :param y_true: etiquetas reales
        :param y_pred: etiquetas predichas
        :param output_path: ruta donde se guarda el png
        :return: la ruta del archivo generado
        """
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
        plt.title("Matriz de Confusión - Salary")
        plt.xlabel("Predicción")
        plt.ylabel("Real")
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        return output_path

    @staticmethod
    def plot_roc_curve(model, X_test, y_test, output_path="roc_curve.png"):
        """Guarda la curva ROC como imagen.

        :param model: modelo entrenado con predict_proba
        :param X_test: features de test (ya transformadas)
        :param y_test: etiquetas de test
        :param output_path: ruta donde se guarda el png
        :return: la ruta del archivo generado
        """
        plt.figure(figsize=(6, 5))
        RocCurveDisplay.from_estimator(model, X_test, y_test)
        plt.title("Curva ROC - Salary")
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        return output_path
