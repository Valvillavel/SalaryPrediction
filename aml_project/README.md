# AML Salary Prediction

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Modelo de clasificación binaria (`<=50K` vs `>50K`) sobre el dataset Adult/Salary. Generado a partir del template **cookiecutter-data-science**, refactorizado a **POO**, con pruebas automatizadas (**pytest**) y pipeline reproducible versionado con **DVC**.

## Estado del proyecto

- [x] Exploración inicial en notebook (`notebooks/PrediccionSalario.ipynb`)
- [x] Refactorización a POO (`src/`)
- [x] Suite de pruebas unitarias e integración (`tests/`)
- [x] Pipeline reproducible con DVC (`dvc.yaml`, `params.yaml`)
- [x] Datos y modelos versionados fuera de Git (remote de Google Drive)

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
├── notebooks/
│   └── PrediccionSalario.ipynb    # exploración original, ya no es fuente de verdad
├── references/                    # diccionarios de datos y material explicativo
├── reports/
│   ├── figures/                   # gráficos de EDA y evaluación (output de la etapa `evaluate`)
│   └── metrics.json               # métricas de los 3 modelos (trackeado como metric de DVC)
├── src/
│   ├── config.py                  # rutas y constantes centralizadas
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
│   ├── stages/                    # scripts de entrada por etapa de DVC
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

> **Nota de mantenimiento:** varios módulos de `src/` siguen en camelCase (`dataLoader.py`, `edaAnalyzer.py`, `salaryPreprocessor.py`, `baseModel.py`, etc.), mientras que `src/stages/` ya sigue snake_case. Se recomienda unificar a snake_case en una futura limpieza para evitar errores de import como los que se corrigieron durante la implementación de pytest (`ModuleNotFoundError`, imports relativos rotos).

---

## Instalación

```bash
git clone -b alex https://github.com/Valvillavel/SalatyPrediction.git
cd SalatyPrediction/aml_project

python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/Mac

pip install -r requirements.txt
```

### Traer los datos y modelos versionados con DVC

```bash
dvc pull
```

> El remote configurado es Google Drive (`gdrive://1evpUjsWTrHQA5Z-pfe0QJmECixztztKk`). Las credenciales OAuth (`gdrive_client_id`, `gdrive_client_secret`) deben configurarse localmente y **nunca** commitearse en texto plano — usar `dvc remote modify --local storage gdrive_client_id <valor>` en vez de editar `.dvc/config` directamente.

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

## Ejecutar el proyecto

### Opción A — pipeline reproducible con DVC (recomendado)

```bash
dvc repro
```

Ejecuta únicamente las etapas afectadas por cambios en datos, código o `params.yaml`. Grafo de dependencias:

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
| Neural Network | 0.860 | 0.752 | 0.627 | 0.684 | 0.915 |

- **Random Forest** tiene el mejor AUC-ROC (0.916), muy cerca de la red neuronal (0.915).
- La **Red Neuronal** tiene la mejor Accuracy y Precision, pero el Recall más bajo de los tres — clasifica menos falsos positivos de la clase `>50K`, a costa de identificar menos casos reales de esa clase.
- **Logistic Regression** y **Random Forest** priorizan Recall sobre Precision, con F1-Score prácticamente empatado entre los tres modelos (~0.68).

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
- Ver `requirements.txt` para dependencias completas del entorno (`scikit-learn`, `tensorflow`, `pandas`, `dvc`, `pytest`, entre otras)

---

## Notas de mantenimiento

- El notebook `notebooks/PrediccionSalario.ipynb` se conserva solo como referencia exploratoria; el código de producción vive en `src/`.
- No trackear manualmente con `git add` ningún archivo `.csv`, `.pkl` o `.keras` dentro de `data/` o `models/` — todos se gestionan vía DVC (`dvc add` para datos crudos, `outs` de `dvc.yaml` para artefactos generados por el pipeline).
- Existe un archivo `models/modelo_salary_rf.pkl` que no está referenciado en `dvc.yaml` — revisar si es un artefacto obsoleto de una corrida anterior y, de ser así, eliminarlo para no confundir el tracking de DVC.
- Antes de hacer commit de un cambio en `params.yaml` o en la lógica de alguna etapa, correr `dvc repro` y `pytest -m "not slow"` para confirmar que todo sigue reproducible.
- Pendiente: unificar la convención de nombres de archivos en `src/` (camelCase → snake_case) para consistencia con `src/stages/`.