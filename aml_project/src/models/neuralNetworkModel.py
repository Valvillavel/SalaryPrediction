"""Modelo de Redes Neuronales"""

import numpy as np
from keras import layers, callbacks, Sequential, metrics, models
from .baseModel import BaseModel


class NeuralNetworkModel(BaseModel):
    def __init__(self, input_dim):
        super().__init__("Neural Network")
        self.model = Sequential([
            layers.Dense(64, activation="relu", input_shape=(input_dim,)),
            layers.Dropout(0.3),
            layers.Dense(32, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(1, activation="sigmoid"),
        ])
        self.model.compile(
            optimizer="adam", loss="binary_crossentropy",
            metrics=["accuracy", metrics.Precision(),
                     metrics.Recall(), metrics.AUC(name="auc")],
        )
        self.history = None

    def fit(self, X_train, y_train_bin, epochs=100, batch_size=128):
        early_stop = callbacks.EarlyStopping(monitor="val_loss", patience=10,
                                             restore_best_weights=True)
        self.history = self.model.fit(
            X_train, y_train_bin, validation_split=0.2, epochs=epochs,
            batch_size=batch_size, callbacks=[early_stop], verbose=1,
        )
        return self

    def predict(self, X_test, threshold=0.5):
        proba = self.predict_proba(X_test)
        return (proba >= threshold).astype(int)

    def predict_proba(self, X_test):
        return self.model.predict(X_test).ravel()

    def save(self, path):
        self.model.save(path)

    def load(self, path):
        self.model = models.load_model(path)
        return self
