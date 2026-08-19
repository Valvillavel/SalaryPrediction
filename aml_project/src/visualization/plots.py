"""Graficador"""

import matplotlib.pyplot as plt
import seaborn as sns


class PlotHelper:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        sns.set_style("whitegrid")
        plt.rcParams["figure.figsize"] = (12, 6)

    def _save(self, filename):
        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=150)
        plt.close()

    def countplot(self, df, col, filename):
        sns.countplot(data=df, x=col)
        self._save(filename)

    def confusion_matrix(self, cm, labels, title, filename):
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=labels, yticklabels=labels)
        plt.title(f"Matriz de Confusión - {title}")
        self._save(filename)

    # ... boxplot_by_target, countplot_by_target, heatmap, histograms, roc_curve, etc.
