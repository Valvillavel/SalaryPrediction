"""Centraliza métricas, matriz de confusión, curva ROC y comparación entre modelos"""

from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve,
                             confusion_matrix, classification_report)


class ModelEvaluator:
    def __init__(self, pos_label=" >50K", plotter=None):
        self.pos_label = pos_label
        self.plotter = plotter
        self.results = {}

    def evaluate(self, name, y_true, y_pred, y_true_bin, y_proba):
        metrics = {
            "Accuracy": accuracy_score(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred, pos_label=self.pos_label),
            "Recall": recall_score(y_true, y_pred, pos_label=self.pos_label),
            "F1-Score": f1_score(y_true, y_pred, pos_label=self.pos_label),
            "AUC-ROC": roc_auc_score(y_true_bin, y_proba),
        }
        self.results[name] = metrics
        return metrics

    def confusion_matrix(self, name, y_true, y_pred, labels, filename):
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        if self.plotter:
            self.plotter.confusion_matrix(cm, labels, name, filename)
        return cm

    def roc_curve(self, name, y_true_bin, y_proba):
        return roc_curve(y_true_bin, y_proba)

    def comparison_table(self):
        import pandas as pd
        return pd.DataFrame(self.results).T

    def classification_report(self, y_true, y_pred):
        return classification_report(y_true, y_pred)
