import math

import numpy as np
from tqdm import trange

from models.base import Model
from models.tree import TreeClassifier


class ADABoosting(Model):
    def __init__(self, model, iterations=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model
        self.iterations = iterations
        self.validate_model()
        self.w = None
        self.X, self.y = None, None
        self.model_instances = []
        self.alpha = []

    def fit(self, X, y, *args, **kwargs):
        self.X, self.y = X, y
        self.validate_labels()
        # Initialize weights
        self.w = np.ones(self.X.shape[0]) / self.X.shape[0]

        # Start boosting
        for _ in trange(self.iterations, desc='ADA Iteration'):
            X, y = self.__get_sample_data__()
            self.model_instances.append(self.model(max_depth=1, no_trange=True).fit(X, y))
            y_hat = self.model_instances[-1].predict(X)

            error = ((self.w * (y_hat != y)).sum()) / self.w.sum()
            alpha = np.log((1 - error) / error) if error != 0 else 1
            self.alpha.append(alpha)
            self.w = self.w * np.exp((alpha * (y_hat != y)))

            # Stop when over fitting
            if error == 0:
                break

        # Convert alpha to np array
        self.alpha = np.array(self.alpha)
        assert self.alpha.shape[0] == len(self.model_instances)

    def __predict__(self, X):
        return np.sign(np.array([
            alpha * model.__predict__(X)
            for alpha, model in zip(self.alpha, self.model_instances)
        ]).sum())

    def validate_model(self):
        if not issubclass(self.model, TreeClassifier):
            raise Exception("Model not class of DecisionTree!")

    def __get_sample_data__(self):
        self.w = self.w / self.w.sum()
        indices = np.random.choice(self.X.shape[0], size=self.X.shape[0], p=self.w)
        return self.X[indices, :], self. y[indices]

    def validate_labels(self):
        if (label_count := len(set(self.y))) != 2:
            raise Exception(f"Number of unique labels is not 2, its {label_count}")

    def score(self, X, y, *args, **kwargs):
        y_hats = self.predict(X)
        return (y_hats == y).mean()
