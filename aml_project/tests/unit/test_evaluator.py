import numpy as np
from src.models.modelEvaluator import ModelEvaluator


def test_evaluate_returns_all_expected_metrics():
    evaluator = ModelEvaluator(pos_label=" >50K")
    y_true = np.array([" <=50K", " >50K", " >50K", " <=50K"])
    y_pred = np.array([" <=50K", " >50K", " <=50K", " <=50K"])
    y_true_bin = np.array([0, 1, 1, 0])
    y_proba = np.array([0.1, 0.8, 0.4, 0.2])

    metrics = evaluator.evaluate(
        "Modelo Dummy", y_true, y_pred, y_true_bin, y_proba)
    expected_keys = {"Accuracy", "Precision", "Recall", "F1-Score", "AUC-ROC"}
    assert set(metrics.keys()) == expected_keys
    assert 0 <= metrics["Accuracy"] <= 1


def test_comparison_table_has_one_row_per_model():
    evaluator = ModelEvaluator(pos_label=" >50K")
    y_true_bin = np.array([0, 1, 1, 0])
    y_proba = np.array([0.1, 0.8, 0.4, 0.2])
    y_true = np.array([" <=50K", " >50K", " >50K", " <=50K"])
    y_pred = y_true.copy()

    evaluator.evaluate("Modelo A", y_true, y_pred, y_true_bin, y_proba)
    evaluator.evaluate("Modelo B", y_true, y_pred, y_true_bin, y_proba)

    table = evaluator.comparison_table()
    assert list(table.index) == ["Modelo A", "Modelo B"]


def test_confusion_matrix_shape():
    evaluator = ModelEvaluator(pos_label=" >50K")
    y_true = [" <=50K", " >50K", " >50K", " <=50K"]
    y_pred = [" <=50K", " >50K", " <=50K", " <=50K"]
    cm = evaluator.confusion_matrix("Modelo Dummy", y_true, y_pred,
                                    labels=[" <=50K", " >50K"], filename="dummy.png")
    assert cm.shape == (2, 2)
