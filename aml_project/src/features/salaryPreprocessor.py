"""Encapsula el `ColumnTransformer` con la misma interfaz `fit`/`transform` de scikit-learn,
más persistencia con `joblib`.
"""
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


class SalaryPreprocessor:
    def __init__(self, target_col, test_size=0.30, random_state=42):
        self.target_col = target_col
        self.test_size = test_size
        self.random_state = random_state
        self.preprocessor = None
        self.num_cols = None
        self.cat_cols = None

    def split(self, df):
        y = df[self.target_col]
        X = df.drop(columns=[self.target_col])
        self.num_cols = X.select_dtypes(
            include=["int64", "float64"]).columns.tolist()
        self.cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
        return train_test_split(X, y, test_size=self.test_size,
                                random_state=self.random_state, stratify=y)

    def _build_transformer(self):
        num_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        cat_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])
        self.preprocessor = ColumnTransformer([
            ("num", num_pipeline, self.num_cols),
            ("cat", cat_pipeline, self.cat_cols),
        ])
        return self.preprocessor

    def fit_transform(self, X_train, X_test):
        self._build_transformer()
        self.preprocessor.fit(X_train)
        return self.preprocessor.transform(X_train), self.preprocessor.transform(X_test)

    def get_feature_names(self):
        cat_names = self.preprocessor.named_transformers_["cat"] \
            .named_steps["onehot"].get_feature_names_out(self.cat_cols)
        return self.num_cols + list(cat_names)

    def save(self, path):
        joblib.dump(self.preprocessor, path)

    def load(self, path):
        self.preprocessor = joblib.load(path)
        return self.preprocessor

    def save_processed_datasets(self, X_train, X_test, y_train, y_test, out_dir):
        cols = self.get_feature_names()
        pd.DataFrame(X_train, columns=cols).to_csv(
            out_dir / "X_train_processed.csv", index=False)
        pd.DataFrame(X_test, columns=cols).to_csv(
            out_dir / "X_test_processed.csv", index=False)
        y_train.to_csv(out_dir / "y_train.csv", index=False)
        y_test.to_csv(out_dir / "y_test.csv", index=False)
