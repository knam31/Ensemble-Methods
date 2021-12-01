from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_wine, load_breast_cancer
from sklearn.ensemble import RandomForestClassifier as skBaggingClassifier
from matplotlib import pyplot as plt

from models.tree import TreeClassifier
from models.boosting import ADABoosting

# Pair of models to compare and dataset to you
compare_models = [
    (
        ADABoosting(model=TreeClassifier, iterations=10),
        skBaggingClassifier(n_estimators=10, max_depth=10, max_features=2),
        load_breast_cancer(return_X_y=True),
    ),
]


for model_1, model_2, data in compare_models:
    # Get dataset and split into train and test
    X, y = data
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

    # Get score from model 1
    score_1 = make_pipeline(
        MinMaxScaler(),
        model_1
    ).fit(X_train, y_train).score(X_test, y_test)

    # Get score from model 2
    score_2 = make_pipeline(
        MinMaxScaler(),
        model_2
    ).fit(X_train, y_train).score(X_test, y_test)

    # Plot scores
    plt.barh([
        f"Implemented {str(model_1.__class__.__name__)}", f"Inbuilt {str(model_2.__class__.__name__)}"
    ], [score_1, score_2], height=[0.2, 0.2], edgecolor='k')
    for i, v in enumerate([score_1, score_2]):
        plt.text(v + 0.005, i, str(round(v, 2)))
    plt.show()
