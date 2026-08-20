# Integrar tu proyecto POO (aml_prediction
) con el flujo del Tutorial DVC

## Diferencia clave entre lo que ya armamos y el tutorial

| | Lo que armamos antes | Lo que pide el tutorial DVC |
|---|---|---|
| Carpeta del código | `aml_prediction
/` (paquete instalable, `pip install -e .`) | `src/data/` y `src/models/` (scripts sueltos) |
| Config de hiperparámetros | `config.py` (constantes en Python) | `params.yaml` (leído con `yaml.safe_load`) |
| Orquestación del pipeline | vos corrías `python -m aml_prediction
.modeling.train` a mano | `dvc.yaml` define las etapas y las corre `dvc repro` |
| Tracking de resultados | MLflow (`mlflow.log_metric`) | `metrics/eval.json` + `metrics/plots.csv`, leídos con `dvc metrics show` |
| Versionado de datos/modelos | — | DVC (`dvc add`, remote en Google Drive) |

**No hay que tirar el trabajo de POO.** La solución es: tus clases (`SalaryDataset`,
`build_preprocessor`, `RandomForestClassifier` entrenado, etc.) siguen viviendo en
`aml_prediction
/` como el "motor" del proyecto. Los archivos que pide el tutorial
(`src/data/make_dataset.py`, `src/models/train_model.py`) son **scripts finitos que solo
llaman a tus clases** y le hablan a `params.yaml` y `dvc.yaml`. Es la misma idea de
modularidad de la diapo del profe, un nivel más arriba: el paquete POO es la librería,
los scripts de `src/` son la interfaz de línea de comandos que usa DVC.

---

## 1. Árbol final del proyecto (fusionando ambos)

```text
salary-classifier/
├── .git/
├── .dvc/
│   └── config                     <- remote de Google Drive
├── .gitignore
├── params.yaml                    <- hiperparámetros (rastreado por Git)
├── dvc.yaml                       <- pipeline (rastreado por Git)
├── dvc.lock                       <- generado por dvc repro
├── pyproject.toml
├── requirements.txt
│
├── data/
│   ├── raw/
│   │   ├── salary.csv.dvc          <- puntero (Git)
│   │   └── .gitignore              <- ignora salary.csv real
│   ├── interim/
│   └── processed/
│       └── clean_salary.csv        <- salida de la etapa "preprocess"
│
├── metrics/
│   ├── eval.json                    <- métricas de la etapa "train"
│   └── plots.csv                    <- predicho vs. real, para dvc plots
│
├── models/
│   ├── modelo_salary_rf.pkl.dvc      <- versionado con DVC (pesa mucho)
│   └── preprocessor.pkl.dvc
│
├── aml_prediction
/                <- TU PAQUETE POO (esto ya lo tenías)
│   ├── __init__.py
│   ├── config.py
│   ├── dataset.py                     <- clase SalaryDataset
│   ├── features.py                    <- build_preprocessor, balance_with_smote
│   ├── plots.py
│   └── modeling/
│       ├── __init__.py
│       ├── train.py                    <- lógica de entrenamiento reutilizable
│       └── predict.py
│
├── src/                              <- CAPA FINA que pide el tutorial/DVC
│   ├── data/
│   │   └── make_dataset.py            <- llama a aml_prediction
.dataset
│   └── models/
│       └── train_model.py             <- llama a aml_prediction
.modeling.train
│
├── notebooks/
│   └── 1.0-eda-salary.ipynb
│
└── tests/
    ├── test_dataset.py
    └── test_features.py
```

`src/` no reemplaza a `aml_prediction
/`: es una capa de scripts ejecutables que
`dvc.yaml` puede invocar con `python src/data/make_dataset.py`, mientras que la lógica de
verdad (clases, docstrings, encapsulación) queda en el paquete que ya armamos.

---

## 2. `params.yaml` — adaptado a salary

```yaml
prepare:
  split_ratio: 0.2
  random_state: 42

train:
  n_estimators: 200
  max_depth: 12
  target_col: "salary"
```

---

## 3. `src/data/make_dataset.py` — capa fina sobre tu clase `SalaryDataset`

En vez de reescribir la limpieza acá (como hace el tutorial con `df.dropna()` suelto),
reutilizamos tu clase POO:

```python
# src/data/make_dataset.py
"""Script de preparación de datos: capa fina sobre aml_prediction
.dataset."""

import os
import yaml

from aml_prediction
.dataset import SalaryDataset
from aml_prediction
 import config


def process_data(output_path="data/processed/clean_salary.csv"):
    """Carga y limpia el dataset crudo usando SalaryDataset, y guarda el resultado.

    :param output_path: ruta donde se guarda el csv procesado
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    ds = SalaryDataset(csv_path=config.RAW_SALARY_CSV).load().clean()
    ds.data.to_csv(output_path, index=False)
    print(f"Datos procesados en: {output_path}")


if __name__ == "__main__":
    with open("params.yaml") as f:
        params = yaml.safe_load(f)  # por si en el futuro necesitás algún param acá
    process_data()
```

---

## 4. `src/models/train_model.py` — capa fina sobre `aml_prediction
.modeling.train`

Este es el que se conecta con `metrics/eval.json` y `metrics/plots.csv`, tal como espera
`dvc.yaml`:

```python
# src/models/train_model.py
"""Script de entrenamiento: capa fina sobre aml_prediction
.modeling.train."""

import json
import os

import pandas as pd
import yaml
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from aml_prediction
.features import build_preprocessor, balance_with_smote
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib


def main():
    """Entrena el modelo con los hiperparámetros de params.yaml y registra métricas."""
    with open("params.yaml") as f:
        params = yaml.safe_load(f)

    target_col = params["train"]["target_col"]
    df = pd.read_csv("data/processed/clean_salary.csv")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=params["prepare"]["split_ratio"],
        stratify=y, random_state=params["prepare"]["random_state"],
    )

    preprocessor = build_preprocessor()
    X_train_t = preprocessor.fit_transform(X_train)
    X_test_t = preprocessor.transform(X_test)

    X_train_bal, y_train_bal = balance_with_smote(X_train_t, y_train)

    clf = RandomForestClassifier(
        n_estimators=params["train"]["n_estimators"],
        max_depth=params["train"]["max_depth"],
        random_state=params["prepare"]["random_state"],
        n_jobs=-1,
    )
    clf.fit(X_train_bal, y_train_bal)

    preds = clf.predict(X_test_t)
    metrics = {
        "accuracy": float(accuracy_score(y_test, preds)),
        "precision": float(precision_score(y_test, preds, pos_label=">50K", zero_division=0)),
        "recall": float(recall_score(y_test, preds, pos_label=">50K", zero_division=0)),
        "f1_score": float(f1_score(y_test, preds, pos_label=">50K", zero_division=0)),
    }

    os.makedirs("metrics", exist_ok=True)
    with open("metrics/eval.json", "w") as f:
        json.dump(metrics, f, indent=4)

    pd.DataFrame({"actual": y_test.values, "predicted": preds}).to_csv(
        "metrics/plots.csv", index=False
    )

    os.makedirs("models", exist_ok=True)
    joblib.dump(clf, "models/modelo_salary_rf.pkl")
    joblib.dump(preprocessor, "models/preprocessor.pkl")

    print("Entrenamiento completado. Métricas en metrics/eval.json")


if __name__ == "__main__":
    main()
```

> Nota: `dvc.yaml` necesita que estos scripts corran como **proceso independiente**
> (`python src/models/train_model.py`), por eso acá sí repetimos algo de lógica en vez de
> encadenar clases con `.load().clean()` importando todo — es el precio de que DVC pueda
> versionar cada etapa por separado. Si preferís, podés mover el bloque de entrenamiento
> a `aml_prediction
/modeling/train.py` como una función y que este script solo la llame
> con los parámetros de `params.yaml`; funciona igual, es más DRY.

---

## 5. `dvc.yaml` — pipeline con tus datos

```yaml
stages:
  preprocess:
    cmd: python src/data/make_dataset.py
    deps:
      - src/data/make_dataset.py
      - aml_prediction
      /dataset.py
      - data/raw/salary.csv
    outs:
      - data/processed/clean_salary.csv

  train:
    cmd: python src/models/train_model.py
    deps:
      - src/models/train_model.py
      - aml_prediction
      /features.py
      - data/processed/clean_salary.csv
    params:
      - prepare.split_ratio
      - prepare.random_state
      - train.n_estimators
      - train.max_depth
      - train.target_col
    outs:
      - models/modelo_salary_rf.pkl
      - models/preprocessor.pkl
    metrics:
      - metrics/eval.json:
          cache: false
    plots:
      - metrics/plots.csv:
          template: confusion
          x: predicted
          y: actual
          cache: false
```

Fijate que agregué `aml_prediction
/dataset.py` y `features.py` como `deps`: así, si
modificás una clase del paquete, DVC detecta el cambio y vuelve a correr la etapa
correspondiente en el próximo `dvc repro`. Eso es justamente lo que aporta tu diseño POO
al pipeline de DVC — quedan trazables los cambios en la lógica, no solo en los datos.

---

## 6. Comandos, en orden, igual que el tutorial pero con tu dataset

```bash
# --- Setup ---
pip install "dvc[gdrive]" pandas scikit-learn pyyaml
pip install -e .                     # instala aml_prediction
 en modo editable

git init
dvc init
git add .dvc/ .gitignore
git commit -m "build: initialize DVC in repository"

# --- Remote en Google Drive ---
dvc remote add -d gdrive_remote gdrive://<TU_GDRIVE_FOLDER_ID>
git add .dvc/config
git commit -m "config: set Google Drive as default DVC remote"

# --- Versionar el csv crudo ---
dvc add data/raw/salary.csv
git add data/raw/salary.csv.dvc data/raw/.gitignore
git commit -m "feat(data): track salary.csv with DVC"
dvc push

# --- Correr el pipeline completo ---
dvc repro
dvc metrics show

# --- Registrar la corrida ---
git add dvc.yaml dvc.lock params.yaml src/ metrics/ data/processed/.gitignore
git commit -m "feat(pipeline): complete ML pipeline execution"
dvc push
```

### Comparar experimentos (igual que el tutorial, cambiando `params.yaml`)

```bash
# Editás n_estimators/max_depth en params.yaml, después:
dvc repro
dvc params diff
dvc metrics diff HEAD
git add dvc.lock params.yaml metrics/
git commit -m "experiment: increase n_estimators and max_depth"
dvc push
```

---

## 7. Testing: seguís testeando el paquete, no los scripts de `src/`

Los tests de `pytest` que ya armamos (`tests/test_dataset.py`) siguen funcionando igual,
porque testean las clases de `aml_prediction
/`, no los scripts de `src/`. Eso es otra
ventaja de mantener la lógica en el paquete: `src/data/make_dataset.py` es tan finito que
casi no necesita test propio (a lo sumo, uno de integración que corra `dvc repro` sobre
un dataset chico y verifique que `metrics/eval.json` se generó).

```python
# tests/test_pipeline_integration.py
import json
import subprocess


def test_dvc_repro_genera_metricas(tmp_path, monkeypatch):
    """Test de integración: corre el pipeline y valida que se generen las métricas."""
    # (requiere un data/raw/salary.csv chico de prueba y dvc.yaml apuntando ahí)
    result = subprocess.run(["dvc", "repro"], capture_output=True, text=True)
    assert result.returncode == 0
    with open("metrics/eval.json") as f:
        metrics = json.load(f)
    assert "f1_score" in metrics
```

---

## 8. ¿Y MLflow? (lo que instalamos antes)

El tutorial de DVC no usa MLflow, y no hace falta pelearlos entre sí: son complementarios.
Si el profe pide los dos, simplemente agregás el `mlflow.start_run()` **dentro** de
`src/models/train_model.py`, alrededor del mismo bloque de entrenamiento, así cada
`dvc repro` también queda registrado en `mlflow ui`. DVC versiona datos/pipeline;
MLflow compara corridas visualmente. No es redundante, cada uno resuelve un problema
distinto.
