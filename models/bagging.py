import numpy as np
from sklearn.metrics import r2_score

from models.base import Model


class Bagging(Model):
    def __init__(self, model, n_count=1, model_args=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if model_args is None:
            model_args = {}
        self.model = model
        self.n_count = n_count
        self.model_instances = []
        self.X, self.y = None, None
        self.model_args = model_args

    def fit(self, X, y):
        self.X, self.y = X, y
        self.model_instances = [
            self.model(**self.model_args).fit(*self.__get_sample__())
            for _ in range(self.n_count)
        ]

    def __get_sample__(self):
        indices = np.random.randint(self.X.shape[0], size=self.X.shape[0])
        return self.X[indices, :], self.y[indices]

    def __predict__(self, *args, **kwargs):
        predictions = [
            model.__predict__(*args, **kwargs)
            for model in self.model_instances
        ]
        return self.__get_aggregate_prediction__(predictions)

    def score(self, X, y, *args, **kwargs):
        raise Exception("Abstract method score() called")

    def __get_aggregate_prediction__(self, *args, **kwargs):
        raise Exception("Abstract method __get_aggregate_prediction__() called")


class BaggingClassifier(Bagging):
    def __get_aggregate_prediction__(self, predictions, *args, **kwargs):
        return np.bincount(predictions).argmax()

    def score(self, X, y, *args, **kwargs):
        y_hats = self.predict(X)
        return (y_hats == y).mean()


class BaggingRegression(Bagging):
    def __get_aggregate_prediction__(self, predictions, *args, **kwargs):
        return np.array(predictions).mean()

    def score(self, X, y, *args, **kwargs):
        y_hats = self.predict(X)
        return r2_score(y, y_hats)
