"""Modelo de regresion lineal"""

import joblib
from sklearn.linear_model import LogisticRegression
from .baseModel import BaseModel
from src.config import RANDOM_STATE


class LogisticRegressionModel(BaseModel):
    def __init__(self, random_state=RANDOM_STATE):
        super().__init__("Logistic Regression")
        self.model = LogisticRegression(max_iter=1000, class_weight="balanced",
                                        random_state=random_state)

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        return self

    def predict(self, X_test):
        return self.model.predict(X_test)

    def predict_proba(self, X_test):
        return self.model.predict_proba(X_test)[:, 1]

    def save(self, path):
        joblib.dump(self.model, path)

    def load(self, path):
        self.model = joblib.load(path)
        return self
