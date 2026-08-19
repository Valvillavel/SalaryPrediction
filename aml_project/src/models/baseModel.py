"""Clase abstracta evita duplicar el patrón `fit / predict / predict_proba / save / load` en cada modelo, y permite que `ModelEvaluator` trate a los tres de forma polimórfica."""

from abc import ABC, abstractmethod


class BaseModel(ABC):
    def __init__(self, name):
        self.name = name
        self.model = None

    @abstractmethod
    def fit(self, X_train, y_train): ...

    @abstractmethod
    def predict(self, X_test): ...

    @abstractmethod
    def predict_proba(self, X_test): ...

    @abstractmethod
    def save(self, path): ...

    @abstractmethod
    def load(self, path): ...
