"""Modelo de tuneo para el random forest"""

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold


class RandomForestTuner:
    PARAM_DIST = {
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [10, 15, 20, 30, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2", None],
        "class_weight": ["balanced", "balanced_subsample",
                         {0: 1, 1: 2}, {0: 1, 1: 3}, {0: 1, 1: 4}],
    }

    def __init__(self, n_iter=50, cv_folds=5, random_state=42):
        self.random_state = random_state
        base = RandomForestClassifier(random_state=random_state)
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True,
                             random_state=random_state)
        self.search = RandomizedSearchCV(
            estimator=base, param_distributions=self.PARAM_DIST, n_iter=n_iter,
            cv=cv, scoring="roc_auc", n_jobs=-1, random_state=random_state,
            verbose=2, return_train_score=True,
        )
        self.best_model_ = None

    def fit(self, X_train, y_train):
        self.search.fit(X_train, y_train)
        self.best_model_ = self.search.best_estimator_
        return self

    @property
    def best_params(self):
        return self.search.best_params_

    @property
    def best_score(self):
        return self.search.best_score_

    def save_search(self, path):
        joblib.dump(self.search, path)

    def save_best_model(self, path):
        joblib.dump(self.best_model_, path)
