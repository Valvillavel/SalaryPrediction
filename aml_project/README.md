# AML Salary Prediction

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Modelo de clasificación binaria (`<=50K` vs `>50K`) sobre el dataset Adult/Salary. Generado a partir del template **cookiecutter-data-science**, refactorizado a **POO**, con pruebas automatizadas (**pytest**), pipeline reproducible versionado con **DVC**, tracking de experimentos con **MLflow**, y una **API REST** (FastAPI) para servir el modelo entrenado.

## Integrantes

- Garcia Andrade Alex Rafael
- Hualca Yavi Orlando
- Rivera Vidaurre Ivan Sergio
- Verastegui Orozco Raisa
- Villarroel Veliz Valeria

## Estado del proyecto

- [x] Exploración inicial en notebook (`notebooks/PrediccionSalario.ipynb`)
- [x] Refactorización a POO (`src/`)
- [x] Suite de pruebas unitarias e integración (`tests/`)
- [x] Pipeline reproducible con DVC (`dvc.yaml`, `params.yaml`)
- [x] Datos y modelos versionados fuera de Git (remote de Google Drive)
- [x] Tracking de experimentos con MLflow (`src/tracking.py`)
- [x] API REST de inferencia con FastAPI (`src/api/`)

---

## Estructura del proyecto

```
aml_project/
├── data/
│   ├── external/                  # datos de terceros (vacío por ahora)
│   ├── interim/                   # datos intermedios (vacío por ahora)
│   ├── processed/                 # X_train/X_test/y_train/y_test procesados (output de `preprocess`)
│   └── raw/
│       └── salary.csv             # versionado con DVC (salary.csv.dvc), no está en Git directo
├── docs/                          # proyecto mkdocs (ver docs/mkdocs.yml)
├── models/
│   ├── preprocessor.pkl
│   ├── logreg_model.pkl
│   ├── rf_model.pkl
│   ├── neural_network_model.keras
│   ├── random_forest_tuned.pkl
│   └── random_search_rf.pkl
├── mlruns/                        # tracking store de MLflow (backend de artefactos)
├── mlflow.db                      # backend de metadatos de MLflow (SQLite)
├── notebooks/
│   └── PrediccionSalario.ipynb    # exploración original, ya no es fuente de verdad
├── references/                    # diccionarios de datos y material explicativo
├── reports/
│   ├── figures/                   # gráficos de EDA y evaluación (output de la etapa `evaluate`)
│   └── metrics.json               # métricas de los 3 modelos (trackeado como metric de DVC)
├── src/
│   ├── config.py                  # rutas, constantes y configuración de MLflow
│   ├── tracking.py                # helper start_run() para MLflow, usado por los stages
│   ├── api/
│   │   ├── main.py                # app FastAPI (endpoints /health, /predict, /predict/batch)
│   │   └── schemas.py             # esquemas Pydantic de entrada/salida
│   ├── data/
│   │   └── dataLoader.py          # DataLoader
│   ├── features/
│   │   ├── edaAnalyzer.py         # EDAAnalyzer
│   │   └── salaryPreprocessor.py  # SalaryPreprocessor
│   ├── models/
│   │   ├── baseModel.py           # BaseModel (clase abstracta)
│   │   ├── logisticRegressionModel.py
│   │   ├── randomForestModel.py
│   │   ├── neuralNetworkModel.py
│   │   ├── modelEvaluator.py      # ModelEvaluator
│   │   └── randomForestTuner.py   # RandomForestTuner
│   ├── visualization/
│   │   └── plots.py               # PlotHelper
│   ├── stages/                    # scripts de entrada por etapa de DVC (todos con tracking MLflow)
│   │   ├── preprocess.py
│   │   ├── train_logreg.py
│   │   ├── train_rf.py
│   │   ├── train_nn.py
│   │   ├── evaluate.py
│   │   └── tune_rf.py
│   ├── pipeline.py                # SalaryPredictionPipeline (orquestador)
│   └── main.py                    # punto de entrada manual
├── tests/
│   ├── conftest.py                # fixtures compartidas (backend Agg, tmp_project_dirs, etc.)
│   ├── fixtures/
│   │   └── sample_salary.csv      # dataset reducido para pruebas
│   ├── generateSampleSalary.py    # regenera sample_salary.csv
│   ├── unit/                      # una prueba por clase
│   └── integration/
│       └── test_pipeline.py       # pipeline completo de punta a punta (marcado `slow`)
├── dvc.yaml                       # definición del pipeline (6 etapas)
├── dvc.lock                       # hashes de reproducibilidad (autogenerado)
├── params.yaml                    # hiperparámetros versionados
├── pyproject.toml                 # metadata del paquete + config de herramientas (black, etc.)
├── pytest.ini
├── requirements.txt
├── Makefile                       # comandos de conveniencia (make data, make train, etc.)
└── LICENSE
```

> **Nota de mantenimiento:** varios módulos de `src/` siguen en camelCase (`dataLoader.py`, `edaAnalyzer.py`, `salaryPreprocessor.py`, `baseModel.py`, etc.), mientras que `src/stages/` y `src/api/` ya siguen snake_case. Se recomienda unificar a snake_case en una futura limpieza.

---

## Instalación

```bash
git clone https://github.com/Valvillavel/SalaryPrediction.git
cd SalaryPrediction/aml_project

python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/Mac

pip install -r requirements.txt
```

### Traer los datos y modelos versionados con DVC

```bash
dvc pull
```

> El remote configurado es Google Drive. Las credenciales OAuth (`gdrive_client_id`, `gdrive_client_secret`) deben configurarse localmente y **nunca** commitearse en texto plano — usar `dvc remote modify --local storage gdrive_client_id <valor>` en vez de editar `.dvc/config` directamente.

---

## Arquitectura del código (POO)

| Clase | Módulo | Responsabilidad |
|---|---|---|
| `DataLoader` | `src/data/dataLoader.py` | Carga y validación básica del CSV crudo |
| `EDAAnalyzer` | `src/features/edaAnalyzer.py` | Distribución de variables, correlación, detección de outliers |
| `SalaryPreprocessor` | `src/features/salaryPreprocessor.py` | Split train/test, `ColumnTransformer`, guardado de datasets procesados |
| `BaseModel` | `src/models/baseModel.py` | Contrato común (`fit`, `predict`, `predict_proba`, `save`, `load`) |
| `LogisticRegressionModel`, `RandomForestModel`, `NeuralNetworkModel` | `src/models/*.py` | Implementaciones concretas de `BaseModel` |
| `ModelEvaluator` | `src/models/modelEvaluator.py` | Métricas, matriz de confusión, curva ROC, tabla comparativa |
| `RandomForestTuner` | `src/models/randomForestTuner.py` | `RandomizedSearchCV` sobre Random Forest |
| `PlotHelper` | `src/visualization/plots.py` | Generación y guardado de gráficos en `reports/figures/` |
| `SalaryPredictionPipeline` | `src/pipeline.py` | Orquesta el flujo completo (usado por `src/main.py`) |

---

## Ejecutar el pipeline de entrenamiento

### Opción A — pipeline reproducible con DVC (recomendado)

```bash
dvc repro
```

Ejecuta únicamente las etapas afectadas por cambios en datos, código o `params.yaml`. Cada etapa (`train_logreg`, `train_rf`, `train_nn`, `evaluate`, `tune_rf`) abre automáticamente un run de MLflow — ver sección siguiente.

```bash
dvc dag
```

```
preprocess → train_logreg ─┐
          └→ train_rf ─────┼→ evaluate
          └→ train_nn ─────┘
          └→ tune_rf
```

### Opción B — ejecución directa

```bash
python -m src.main
```

---

## Tracking de experimentos (MLflow)

Todos los stages de entrenamiento y evaluación registran automáticamente parámetros, métricas y modelos en MLflow a través del helper `src/tracking.py::start_run()`. El backend de tracking es un SQLite local (`mlflow.db`), configurado en `src/config.py`:

```python
MLFLOW_TRACKING_URI = "sqlite:///" + str(ROOT_DIR / "mlflow.db")
MLFLOW_EXPERIMENT_NAME = "salary-prediction"
```

Cada run queda etiquetado con el commit de Git y el nombre del stage de DVC que lo generó (`git_commit`, `dvc_stage`), lo que permite correlacionar una corrida de MLflow con el estado exacto del código que la produjo.

Levantar la UI de MLflow para explorar las corridas:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Por defecto queda disponible en `http://127.0.0.1:5000`.

- `train_logreg` y `train_rf` usan `mlflow.sklearn.autolog()` además de loguear el modelo explícitamente.
- `train_nn` loguea `epochs`/`batch_size` (desde `params.yaml`) y el modelo Keras.
- `tune_rf` registra cada combinación probada por `RandomizedSearchCV` como un run anidado (`nested=True`), más los mejores hiperparámetros y el mejor score.
- `evaluate` loguea las métricas de los 3 modelos y adjunta `reports/metrics.json` y las figuras como artefactos.

> **Nota:** `mlruns/` y `mlflow.db` están actualmente versionados directo en Git (no vía DVC), lo que ya pesa ~75 MB en el historial. Si el equipo sigue generando experimentos, conviene mover estos artefactos a `.gitignore` + DVC (o a un tracking server remoto) para no seguir inflando el repositorio.

---

## API REST de inferencia (FastAPI)

`src/api/main.py` expone el modelo Random Forest afinado (`models/random_forest_tuned.pkl`) vía HTTP.

### Levantar el servidor

```bash
uvicorn src.api.main:app --reload
```

Documentación interactiva autogenerada en `http://127.0.0.1:8000/docs`.

### Endpoints

| Método | Ruta | Descripción |
|---|---|---|
| `GET` | `/health` | Estado del servicio y si el modelo se cargó correctamente |
| `POST` | `/predict` | Predicción para una sola persona |
| `POST` | `/predict/batch` | Predicción para una lista de personas |

Ejemplo de request a `/predict`:

```json
{
  "age": 39,
  "workclass": "Private",
  "fnlwgt": 77516,
  "education": "Bachelors",
  "education-num": 13,
  "marital-status": "Never-married",
  "occupation": "Adm-clerical",
  "relationship": "Not-in-family",
  "race": "White",
  "sex": "Male",
  "capital-gain": 0,
  "capital-loss": 0,
  "hours-per-week": 40,
  "native-country": "United-States"
}
```

Respuesta:

```json
{
  "salary_prediction": "<=50K",
  "probability_above_50k": 0.1234,
  "model_used": "random_forest_tuned"
}
```

> El servidor requiere que `models/preprocessor.pkl` y `models/random_forest_tuned.pkl` existan (`dvc pull` o `dvc repro` primero). Si faltan, `/health` reporta `model_loaded: false` y `/predict` responde `503` en vez de fallar de forma críptica.

---

## Parámetros configurables (`params.yaml`)

```yaml
preprocessing:
  test_size: 0.3
  random_state: 42

neural_network:
  epochs: 100
  batch_size: 128

tuner:
  n_iter: 50
  cv_folds: 5
```

Cambiar un valor y correr `dvc repro` re-entrena únicamente las etapas afectadas.

---

## Resultados actuales

Métricas de la última corrida registrada en `reports/metrics.json`:

| Modelo | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|---|---|---|---|---|---|
| Logistic Regression | 0.809 | 0.569 | 0.856 | 0.684 | 0.908 |
| Random Forest | 0.800 | 0.553 | 0.883 | 0.680 | 0.916 |
| Neural Network | 0.861 | 0.740 | 0.652 | 0.693 | 0.916 |

- **Random Forest** y **Neural Network** empatan en AUC-ROC (~0.916), ligeramente por encima de Logistic Regression.
- La **Red Neuronal** tiene la mejor Accuracy, Precision y F1-Score, pero el Recall más bajo — identifica menos casos reales de la clase `>50K` a cambio de menos falsos positivos.
- **Logistic Regression** y **Random Forest** priorizan Recall sobre Precision.
- El modelo servido por la API (`random_forest_tuned.pkl`) es el resultado de `tune_rf`, no el `rf_model.pkl` base — sus métricas específicas no están en `metrics.json` (solo se evalúan los 3 modelos base), pero quedan registradas como `best_cv_roc_auc` en el run de MLflow correspondiente.

Para reproducir o comparar contra una versión anterior:

```bash
dvc metrics show
dvc metrics diff HEAD~1
```

Los gráficos de EDA, matrices de confusión y curvas ROC quedan en `reports/figures/`.

---

## Pruebas

```bash
# Pruebas rápidas (día a día, sin entrenar la red neuronal ni correr el tuner completo)
pytest -m "not slow"

# Suite completa
pytest

# Con reporte de cobertura
pytest --cov=src --cov-report=term-missing
```

La suite está dividida en:
- `tests/unit/` — una prueba por clase, con datos sintéticos (`tests/fixtures/sample_salary.csv`)
- `tests/integration/test_pipeline.py` — pipeline completo de punta a punta (marcado `@pytest.mark.slow`)

Regenerar el dataset de prueba si es necesario:

```bash
python tests/generateSampleSalary.py --n-per-class 25 --output tests/fixtures/sample_salary.csv
```

> Pendiente: no hay pruebas todavía para `src/api/` (endpoints FastAPI) ni para `src/tracking.py`. Se recomienda agregar `tests/unit/test_api.py` con `TestClient` de FastAPI y mockear MLflow en los tests de los stages para no depender de una corrida real.

---

## Versionado de datos y experimentos (DVC)

```bash
dvc pull              # descargar datos/modelos versionados
dvc push              # subir cambios de datos/modelos al remote
dvc exp run --set-param neural_network.epochs=50   # experimento sin ensuciar el historial de Git
dvc exp show           # comparar experimentos corridos
```

---

## Requisitos

- Python 3.11+ (probado también en 3.13)
- Ver `requirements.txt` para dependencias completas del entorno (`scikit-learn`, `tensorflow`, `mlflow`, `fastapi`, `uvicorn`, `dvc`, `pytest`, entre otras)

---

## Notas de mantenimiento

- El notebook `notebooks/PrediccionSalario.ipynb` se conserva solo como referencia exploratoria; el código de producción vive en `src/`.
- No trackear manualmente con `git add` ningún archivo `.csv`, `.pkl` o `.keras` dentro de `data/` o `models/` — todos se gestionan vía DVC (`dvc add` para datos crudos, `outs` de `dvc.yaml` para artefactos generados por el pipeline).
- `mlruns/` y `mlflow.db` (~75 MB) están commiteados directo en Git — evaluar moverlos a `.gitignore` + DVC o a un tracking server remoto antes de que el historial crezca más.
- Antes de hacer commit de un cambio en `params.yaml` o en la lógica de alguna etapa, correr `dvc repro` y `pytest -m "not slow"` para confirmar que todo sigue reproducible.
- Pendiente: unificar la convención de nombres de archivos en `src/` (camelCase → snake_case) para consistencia con `src/stages/` y `src/api/`.
- Pendiente: cobertura de pruebas para `src/api/` y `src/tracking.py`.
