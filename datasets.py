import csv
import pandas as pd
from urllib import request
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import load_breast_cancer

import numpy as np


class Dataset:

    URL = None

    def fetch_url(self):
        return request.urlopen(self.URL)

    def get_data(self):
        return self.fetch_url()

    def load(self):
        data = np.array(list(csv.reader(map(lambda x: x.decode('utf-8'), self.get_data().readlines()))))
        X, y = data[:, 0:-1], data[:, -1]
        X = X.astype(float)
        try:
            y = y.astype(float)
        except ValueError:
            y = LabelEncoder().fit_transform(y)
        return X, y


class Haberman(Dataset):
    URL = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/haberman.csv'


class Sonar(Dataset):
    URL = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/sonar.csv'


class WheatSeeds(Dataset):
    URL = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/wheat-seeds.csv'


class WineQualityWhite(Dataset):
    URL = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/winequality-white.csv'


def load_haberman():
    return Haberman().load()


def load_sonar():
    return Sonar().load()


def load_wheat():
    return WheatSeeds().load()


def load_wine_quality():
    return WineQualityWhite().load()

