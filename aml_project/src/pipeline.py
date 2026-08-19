"""pipeline"""

from src import config
from src.data.dataLoader import DataLoader
from src.features.salaryPreprocessor import SalaryPreprocessor
from src.features.salaryPreprocessor import EDAAnalyzer
from src.visualization.plots import PlotHelper
from src.models.logisticRegressionModel import LogisticRegressionModel
from src.models.randomForestModel import RandomForestModel
from src.models.neuralNetworkModel import NeuralNetworkModel
from src.models.modelEvaluator import ModelEvaluator
from src.models.randomForestTuner import RandomForestTuner


class SalaryPredictionPipeline:
    def __init__(self):
        self.plotter = PlotHelper(config.FIGURES_DIR)
        self.preprocessor = SalaryPreprocessor(config.TARGET_COLUMN)
        self.evaluator = ModelEvaluator(
            pos_label=config.POSITIVE_LABEL, plotter=self.plotter)

    def run(self):
        df = DataLoader(config.RAW_DATA_PATH).load()

        eda = EDAAnalyzer(df, config.TARGET_COLUMN,
                          self.preprocessor.num_cols, self.preprocessor.cat_cols, self.plotter)

        X_train, X_test, y_train, y_test = self.preprocessor.split(df)
        X_train_p, X_test_p = self.preprocessor.fit_transform(X_train, X_test)
        self.preprocessor.save(config.MODELS_DIR / "preprocessor.pkl")
        self.preprocessor.save_processed_datasets(X_train_p, X_test_p, y_train, y_test,
                                                  config.PROCESSED_DATA_DIR)

        y_train_bin = (y_train == config.POSITIVE_LABEL).astype(int)
        y_test_bin = (y_test == config.POSITIVE_LABEL).astype(int)

        models = [
            LogisticRegressionModel(),
            RandomForestModel(),
        ]
        for m in models:
            m.fit(X_train_p, y_train)
            y_pred = m.predict(X_test_p)
            y_proba = m.predict_proba(X_test_p)
            self.evaluator.evaluate(
                m.name, y_test, y_pred, y_test_bin, y_proba)
            m.save(config.MODELS_DIR /
                   f"{m.name.lower().replace(' ', '_')}.pkl")

        nn = NeuralNetworkModel(input_dim=X_train_p.shape[1])
        nn.fit(X_train_p, y_train_bin)
        nn_proba = nn.predict_proba(X_test_p)
        nn_pred_labels = [config.POSITIVE_LABEL if p >=
                          0.5 else " <=50K" for p in nn_proba]
        self.evaluator.evaluate(
            nn.name, y_test, nn_pred_labels, y_test_bin, nn_proba)
        nn.save(config.MODELS_DIR / "neural_network_model.keras")

        print(self.evaluator.comparison_table())

        tuner = RandomForestTuner()
        tuner.fit(X_train_p, y_train)
        tuner.save_best_model(config.MODELS_DIR / "random_forest_tuned.pkl")
        tuner.save_search(config.MODELS_DIR / "random_search_rf.pkl")
        print("Mejores hiperparámetros:", tuner.best_params)
