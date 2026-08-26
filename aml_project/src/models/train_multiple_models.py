"""Script para entrenar múltiples modelos con diferentes configuraciones."""

import pandas as pd
import yaml
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from aml_prediction.dataset import SalaryDataset
from aml_prediction.features import build_preprocessor, balance_with_smote
from aml_prediction.tracking.experiment_tracker import MLflowExperimentTracker
from aml_prediction import config


def load_and_prepare_data():
    """Carga y prepara los datos una sola vez."""
    import numpy as np
    
    print("📊 Cargando y preparando datos...")
    
    # Cargar datos limpios
    ds = SalaryDataset(csv_path=config.RAW_SALARY_CSV).load().clean()
    
    # Convertir pd.NA a np.nan para compatibilidad con sklearn
    ds._df = ds._df.fillna(np.nan)
    
    # Split
    with open("params.yaml") as f:
        params = yaml.safe_load(f)
    
    X_train, X_test, y_train, y_test = ds.train_test_split(
        test_size=params["prepare"]["split_ratio"]
    )
    
    # Convertir target a 0/1 (los strings ya están sin espacios después de clean())
    y_train = y_train.map({"<=50K": 0, ">50K": 1})
    y_test = y_test.map({"<=50K": 0, ">50K": 1})
    
    # Preprocesar features
    preprocessor = build_preprocessor()
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)
    
    # Aplicar SMOTE
    X_train_balanced, y_train_balanced = balance_with_smote(
        X_train_transformed, y_train
    )
    
    print(f"✅ Datos preparados: {X_train_balanced.shape[0]} muestras entrenamiento, {X_test_transformed.shape[0]} test")
    
    return X_train_balanced, X_test_transformed, y_train_balanced, y_test, preprocessor, params


def train_model(model_name, model, params, X_train, y_train, X_test, y_test, 
                preprocessor, run_name, experiment_name="aml_prediction"):
    """Entrena un modelo y registra en MLflow."""
    
    print(f"\n🔬 Entrenando {model_name} - {run_name}...")
    
    tracker = MLflowExperimentTracker(experiment_name=experiment_name)
    
    tracker.log_run(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        params=params,
        preprocessor=preprocessor,
        run_name=run_name,
        dvc_stage="train_multiple",
        register_as=None,  # No registrar automáticamente
    )
    
    # Entrenar y evaluar manualmente para mostrar resultados
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, pos_label=1)
    precision = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
    recall = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
    
    print(f"   📈 Accuracy: {accuracy:.4f}")
    print(f"   📈 F1-Score: {f1:.4f}")
    print(f"   📈 Precision: {precision:.4f}")
    print(f"   📈 Recall: {recall:.4f}")
    
    return accuracy, f1


def main():
    """Entrena 3 modelos diferentes con 2 configuraciones cada uno."""
    
    # Preparar datos una sola vez
    X_train, X_test, y_train, y_test, preprocessor, base_params = load_and_prepare_data()
    
    print("\n" + "="*70)
    print("🎯 ENTRENANDO MÚLTIPLES MODELOS CON MLFLOW")
    print("="*70)
    
    results = []
    
    # ========================================================================
    # MODELO 1: RandomForestClassifier
    # ========================================================================
    print("\n" + "🌲"*35)
    print("MODELO 1: Random Forest Classifier")
    print("🌲"*35)
    
    # Experimento 1.1: Configuración conservadora (menos árboles, más profundidad)
    rf_params_1 = {
        "n_estimators": 100,
        "max_depth": 15,
        "min_samples_split": 5,
        "random_state": config.RANDOM_STATE
    }
    acc1, f1_1 = train_model(
        "RandomForest",
        RandomForestClassifier(**rf_params_1, n_jobs=-1),
        rf_params_1,
        X_train, y_train, X_test, y_test,
        preprocessor,
        "RandomForest_v1_conservative"
    )
    results.append(("RandomForest_v1", acc1, f1_1))
    
    # Experimento 1.2: Configuración agresiva (más árboles, menos profundidad)
    rf_params_2 = {
        "n_estimators": 300,
        "max_depth": 10,
        "min_samples_split": 10,
        "random_state": config.RANDOM_STATE
    }
    acc2, f1_2 = train_model(
        "RandomForest",
        RandomForestClassifier(**rf_params_2, n_jobs=-1),
        rf_params_2,
        X_train, y_train, X_test, y_test,
        preprocessor,
        "RandomForest_v2_aggressive"
    )
    results.append(("RandomForest_v2", acc2, f1_2))
    
    # ========================================================================
    # MODELO 2: Logistic Regression
    # ========================================================================
    print("\n" + "📈"*35)
    print("MODELO 2: Logistic Regression")
    print("📈"*35)
    
    # Experimento 2.1: Regularización L2 (Ridge)
    lr_params_1 = {
        "penalty": "l2",
        "C": 1.0,
        "max_iter": 1000,
        "solver": "lbfgs",
        "random_state": config.RANDOM_STATE
    }
    acc3, f1_3 = train_model(
        "LogisticRegression",
        LogisticRegression(**lr_params_1, n_jobs=-1),
        lr_params_1,
        X_train, y_train, X_test, y_test,
        preprocessor,
        "LogisticRegression_v1_l2"
    )
    results.append(("LogisticRegression_v1", acc3, f1_3))
    
    # Experimento 2.2: Regularización L1 (Lasso)
    lr_params_2 = {
        "penalty": "l1",
        "C": 0.5,
        "max_iter": 1000,
        "solver": "saga",
        "random_state": config.RANDOM_STATE
    }
    acc4, f1_4 = train_model(
        "LogisticRegression",
        LogisticRegression(**lr_params_2, n_jobs=-1),
        lr_params_2,
        X_train, y_train, X_test, y_test,
        preprocessor,
        "LogisticRegression_v2_l1"
    )
    results.append(("LogisticRegression_v2", acc4, f1_4))
    
    # ========================================================================
    # MODELO 3: Gradient Boosting Classifier
    # ========================================================================
    print("\n" + "🚀"*35)
    print("MODELO 3: Gradient Boosting Classifier")
    print("🚀"*35)
    
    # Experimento 3.1: Configuración rápida (pocos estimadores)
    gb_params_1 = {
        "n_estimators": 100,
        "learning_rate": 0.1,
        "max_depth": 5,
        "subsample": 0.8,
        "random_state": config.RANDOM_STATE
    }
    acc5, f1_5 = train_model(
        "GradientBoosting",
        GradientBoostingClassifier(**gb_params_1),
        gb_params_1,
        X_train, y_train, X_test, y_test,
        preprocessor,
        "GradientBoosting_v1_fast"
    )
    results.append(("GradientBoosting_v1", acc5, f1_5))
    
    # Experimento 3.2: Configuración precisa (más estimadores, menor learning rate)
    gb_params_2 = {
        "n_estimators": 200,
        "learning_rate": 0.05,
        "max_depth": 7,
        "subsample": 0.9,
        "random_state": config.RANDOM_STATE
    }
    acc6, f1_6 = train_model(
        "GradientBoosting",
        GradientBoostingClassifier(**gb_params_2),
        gb_params_2,
        X_train, y_train, X_test, y_test,
        preprocessor,
        "GradientBoosting_v2_precise"
    )
    results.append(("GradientBoosting_v2", acc6, f1_6))
    
    # ========================================================================
    # RESUMEN FINAL
    # ========================================================================
    print("\n" + "="*70)
    print("📊 RESUMEN DE RESULTADOS")
    print("="*70)
    print(f"{'Modelo':<30} {'Accuracy':>12} {'F1-Score':>12}")
    print("-"*70)
    
    for model_name, accuracy, f1 in results:
        print(f"{model_name:<30} {accuracy:>12.4f} {f1:>12.4f}")
    
    # Encontrar el mejor modelo
    best_model = max(results, key=lambda x: x[2])  # Ordenar por F1-Score
    print("-"*70)
    print(f"🏆 MEJOR MODELO: {best_model[0]} (F1-Score: {best_model[2]:.4f})")
    print("="*70)
    
    print("\n✅ Todos los experimentos registrados en MLflow.")
    print("💡 Ejecutá 'mlflow ui' para ver los resultados en la interfaz web.")


if __name__ == "__main__":
    main()
