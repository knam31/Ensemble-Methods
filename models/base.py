import numpy as np


class Model:

    def __init__(self, *args, **kwargs):
        pass

    def fit(self, *args, **kwargs):
        raise Exception("Abstract method fit() called")

    def predict(self, X):
        y_hats = []

        for data in X:
            y_hats.append(self.__predict__(data))

        return np.array(y_hats)

    def score(self, X, y, *args, **kwargs):
        raise Exception("Abstract method score() called")

    def __predict__(self, *args, **kwargs):
        raise Exception("Abstract method __predict__() called")

    def error(self, X, y):
        return 1 - self.score(X, y)
