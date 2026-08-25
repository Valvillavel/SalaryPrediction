"""Cada método de la clase reemplaza un bloque del EDA.
Los gráficos se delegan a `PlotHelper` y se guardan en `reports/figures/`.
"""


class EDAAnalyzer:
    def __init__(self, df, target, num_cols, cat_cols, plotter):
        self.df = df
        self.target = target
        self.num_cols = num_cols
        self.cat_cols = cat_cols
        self.plotter = plotter

    def target_distribution(self):
        counts = self.df[self.target].value_counts()
        pct = self.df[self.target].value_counts(normalize=True) * 100
        self.plotter.countplot(self.df, self.target, "distribucion_target.png")
        return counts, pct

    def numeric_summary(self):
        self.plotter.histograms(
            self.df[self.num_cols], "histogramas_numericas.png")
        for col in self.num_cols:
            self.plotter.boxplot_by_target(self.df, self.target, col,
                                           f"boxplot_{col}.png")

    def categorical_summary(self):
        for col in self.cat_cols:
            self.plotter.countplot_by_target(self.df, col, self.target,
                                             f"cat_{col}_vs_target.png")

    def correlation_matrix(self):
        corr = self.df[self.num_cols].corr()
        self.plotter.heatmap(corr, "matriz_correlacion.png")
        return corr

    def detect_outliers_iqr(self) -> dict:
        report = {}
        for col in self.num_cols:
            q1, q3 = self.df[col].quantile([0.25, 0.75])
            iqr = q3 - q1
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            mask = (self.df[col] < lower) | (self.df[col] > upper)
            report[col] = {"count": int(mask.sum()), "pct": mask.mean() * 100}
        return report

    def run_full_eda(self):
        self.target_distribution()
        self.numeric_summary()
        self.categorical_summary()
        self.correlation_matrix()
        return self.detect_outliers_iqr()
