from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris, load_breast_cancer,load_digits,load_diabetes, load_boston, load_wine
import datasets as db
from sklearn.ensemble import RandomForestClassifier as skBaggingClassifier

from models.tree import TreeClassifier, TreeRegression, RandomTreeClassifier,RandomTreeRegression
from models.bagging import BaggingClassifier,BaggingRegression
from models.boosting import ADABoosting


model_and_datasets = [

#     (
#         TreeClassifier(max_depth=10),
#         load_iris(return_X_y=True),
#         'Iris dataset'
#     ),
#     (
#         TreeClassifier(max_depth=10),
#         load_breast_cancer(return_X_y=True),
#         'Wisconsin Breast Cancer Dataset'
#     ),
#     (
#         TreeClassifier(max_depth=10),
#         load_wine(return_X_y=True),
#         'Wine Dataset'
#     ),
#     (
#         TreeClassifier(max_depth=10),
#         load_digits(return_X_y=True),
#         'Digits Dataset'
#     ),
#     (
#         TreeClassifier(max_depth=10),
#         db.load_sonar(),
#         'Sonar Dataset'
#     ),
#     (
#         TreeClassifier(max_depth=10),
#         db.load_haberman(),
#         'Haberman Dataset'
#     ),
#     (
#         TreeClassifier(max_depth=10),
#         db.load_wheat(),
#         'Wheat-seeds Dataset'
#     ),
#     (
#         TreeRegression(max_depth=10),
#         load_diabetes(return_X_y=True),
#         'Diabetes Dataset'
#     ),
#     (
#         TreeRegression(max_depth=10),
#         load_boston(return_X_y=True),
#         'Boston Dataset'
#     ),
#
#     (
#         BaggingClassifier(model=TreeClassifier, n_count=10, model_args={'max_depth': 10, 'min_group_size': 5}),
#         load_iris(return_X_y=True),
#         'Iris dataset'
#     ),
#     (
#         BaggingClassifier(model=TreeClassifier, n_count=10, model_args={'max_depth': 10, 'min_group_size': 5}),
#         load_breast_cancer(return_X_y=True),
#         'Wisconsin Breast Cancer Dataset'
#     ),
#     (
#         BaggingClassifier(model=TreeClassifier, n_count=10, model_args={'max_depth': 10, 'min_group_size': 5}),
#         load_wine(return_X_y=True),
#         'Wine Dataset'
#     ),
#     (
#         BaggingClassifier(model=TreeClassifier, n_count=10, model_args={'max_depth': 10, 'min_group_size': 5}),
#         load_digits(return_X_y=True),
#         'Digits Dataset'
#     ),
#     (
#         BaggingClassifier(model=TreeClassifier, n_count=10, model_args={'max_depth': 10, 'min_group_size': 5}),
#         db.load_sonar(),
#         'Sonar Dataset'
#     ),
#     (
#         BaggingClassifier(model=TreeClassifier, n_count=10, model_args={'max_depth': 10, 'min_group_size': 5}),
#         db.load_haberman(),
#         'Haberman Dataset'
#     ),
#     (
#         BaggingClassifier(model=TreeClassifier, n_count=10, model_args={'max_depth': 10, 'min_group_size': 5}),
#         db.load_wheat(),
#         'Wheat-seeds Dataset'
#     ),
#     (
#         BaggingRegression(model=TreeRegression, n_count=10, model_args={'max_depth': 10, 'min_group_size': 5}),
#         load_diabetes(return_X_y=True),
#         'Diabetes Dataset'
#     ),
#     (
#         BaggingRegression(model=TreeRegression, n_count=10, model_args={'max_depth': 10, 'min_group_size': 5}),
#         load_boston(return_X_y=True),
#         'Boston Dataset'
#     ),
# (
#         BaggingClassifier(model=RandomTreeClassifier, n_count=10, model_args={'m': 2, 'max_depth': 10, 'min_group_size': 5}),
#         load_iris(return_X_y=True),
#         'Iris dataset'
#     ),
#     (
#         BaggingClassifier(model=RandomTreeClassifier, n_count=10, model_args={'m': 6, 'max_depth': 10, 'min_group_size': 5}),
#         load_breast_cancer(return_X_y=True),
#         'Wisconsin Breast Cancer Dataset'
#     ),
#     (
#         BaggingClassifier(model=RandomTreeClassifier, n_count=10, model_args={'m': 4, 'max_depth': 10, 'min_group_size': 5}),
#         load_wine(return_X_y=True),
#         'Wine Dataset'
#     ),
#     (
#         BaggingClassifier(model=RandomTreeClassifier, n_count=10, model_args={'m': 8, 'max_depth': 10, 'min_group_size': 5}),
#         load_digits(return_X_y=True),
#         'Digits Dataset'
#     ),
#     (
#         BaggingClassifier(model=RandomTreeClassifier, n_count=10, model_args={'m': 8, 'max_depth': 10, 'min_group_size': 5}),
#         db.load_sonar(),
#         'Sonar Dataset'
#     ),
#     (
#         BaggingClassifier(model=RandomTreeClassifier, n_count=10, model_args={'m': 2, 'max_depth': 10, 'min_group_size': 5}),
#         db.load_haberman(),
#         'Haberman Dataset'
#     ),
#     (
#         BaggingClassifier(model=RandomTreeClassifier, n_count=10, model_args={'m': 3, 'max_depth': 10, 'min_group_size': 5}),
#         db.load_wheat(),
#         'Wheat-seeds Dataset'
#     ),
#     (
#         BaggingRegression(model=RandomTreeRegression, n_count=10, model_args={'m': 4, 'max_depth': 10, 'min_group_size': 5}),
#         load_diabetes(return_X_y=True),
#         'Diabetes Dataset'
#     ),
#     (
#         BaggingRegression(model=RandomTreeRegression, n_count=10, model_args={'m': 4, 'max_depth': 10, 'min_group_size': 5}),
#         load_boston(return_X_y=True),
#         'Boston Dataset'
#     ),

    (
        ADABoosting(model=TreeClassifier, iterations=30),
        load_breast_cancer(return_X_y=True),
        'Wisconsin Breast Cancer Dataset'
    ),
    (
        ADABoosting(model=TreeClassifier, iterations=30),
        db.load_sonar(),
        'Sonar Dataset'
    ),
    (
        ADABoosting(model=TreeClassifier, iterations=100),
        db.load_haberman(),
        'Haberman Dataset'
    ),
]

for model, data, dataset_name in model_and_datasets:
    # Get dataset and split into train and test
    X, y = data
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

    # Get score from model
    pipe = make_pipeline(MinMaxScaler(), model).fit(X_train, y_train)
    score = pipe.score(X_test, y_test)

    print(f"{model.__class__.__name__} on {dataset_name} error: {round(1-score,4)}")
