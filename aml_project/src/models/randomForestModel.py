"""Modelo de Random Forest"""

import joblib
from sklearn.ensemble import RandomForestClassifier
from .baseModel import BaseModel
from src.config import RANDOM_STATE


class RandomForestModel(BaseModel):
    def __init__(self, random_state=RANDOM_STATE):
        super().__init__("Random Forest")
        self.model = RandomForestClassifier(
            n_estimators=100, max_depth=15, class_weight='balanced', random_state=random_state, n_jobs=-1)

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
