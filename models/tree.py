import numpy as np
from models.base import Model
from sklearn.metrics import r2_score


class Tree(Model):
    def __init__(self, max_depth=3, min_group_size=5, depth=0, data=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_depth = max_depth
        self.min_group_size = min_group_size
        self.X, self.y, self.XY, self.data = None, None, None, data
        self.depth = depth
        self.left, self.right = None, None
        self.condition = None

    def fit(self, X, y):
        # Store initial data
        self.X = X
        self.y = y
        self.XY = np.c_[self.X, self.y]
        self.data = self.XY

        # Grow tree
        self.grow()

        # Return tree
        return self

    def grow(self):
        depth = self.depth + 1
        # Grow if current depth less than max depth and current data in greater than min group size
        if depth <= self.max_depth and self.data.shape[0] > self.min_group_size:
            # get best split, these are the data for left and right tree
            right, left, condition = self.get_best_split()

            # if split exists, grow tree with increasing depths
            if all(map(lambda x: x is not None, [left, right])):
                self.condition = condition
                self.left = self.__class__(self.max_depth, self.min_group_size, depth, left).grow()
                self.right = self.__class__(self.max_depth, self.min_group_size, depth, right).grow()

        return self

    def get_attribute_value_loop(self, index):
        raise Exception("Abstract method get_attribute_value_loop called!")

    def get_attribute_index_loop(self):
        raise Exception("Abstract method get_attribute_index_loop called!")

    def get_best_split(self):
        # Get impurity for current data
        base_impurity = self.get_impurity(self.data)

        # Initialize right and left_split
        right_split, left_split, condition = None, None, None

        # For each attribute index, -1 here because the last entry will be output
        for attr_index in self.get_attribute_index_loop():
            for attr_val in self.get_attribute_value_loop(attr_index):
                # Split data based on attr_val
                left = self.data[self.data[:, attr_index] < attr_val]
                right = self.data[self.data[:, attr_index] >= attr_val]

                # Proceed only if splits have min group size
                if all(map(lambda x: x.shape[0] > self.min_group_size, [left, right])):
                    # Get impurity for left and right data multiply by fraction of split
                    left_impurity = self.get_impurity(left) * (left.shape[0] / self.data.shape[0])
                    right_impurity = self.get_impurity(right) * (right.shape[0] / self.data.shape[0])

                    # If sum of left and right impurity less than base impurity
                    if left_impurity + right_impurity < base_impurity:
                        base_impurity = left_impurity + right_impurity
                        right_split, left_split = right, left
                        # Condition of split
                        condition = (attr_index, attr_val)

        # return splits
        return right_split, left_split, condition

    def __predict__(self, X):
        # Start from base of tree
        tree = self

        # While tree has condition
        while tree.condition:
            index, value = tree.condition
            if X[index] < value:
                tree = tree.left
            else:
                tree = tree.right

        # Raise Exception if leaf tree has condition
        if tree.condition:
            raise Exception("Something went wrong, leaf tree has condition")

        # Return output
        return tree.get_tree_output()

    @staticmethod
    def get_impurity(data):
        raise Exception("Abstract method get_impurity called!")

    def get_tree_output(self):
        raise Exception("Abstract method get_tree_output called!")


class TreeClassifier(Tree):
    @staticmethod
    def get_impurity(data):
        impurity = 0
        labels = set(data[:, -1])

        for label in labels:
            proportion = (np.array(data[:, -1] == label).sum()) / data.shape[0]
            impurity += proportion * proportion

        impurity = 1 - impurity
        return impurity

    def get_tree_output(self):
        return np.bincount(np.array(self.data[:, -1], dtype=int)).argmax()

    def score(self, X, y, *args, **kwargs):
        y_hats = self.predict(X)
        return (y_hats == y).mean()

    def get_attribute_value_loop(self, index):
        return np.arange(self.data[:, index].min(), self.data[:, index].max(), 0.1)

    def get_attribute_index_loop(self):
        return range(self.data.shape[1] - 1)


class TreeRegression(Tree):
    @staticmethod
    def get_impurity(data):
        y_hat = data[:, -1].mean()
        error = (data[:, -1] - y_hat) ** 2
        return error.sum()

    def score(self, X, y, *args, **kwargs):
        y_hats = self.predict(X)
        return r2_score(y, y_hats)

    def get_tree_output(self):
        return self.data[:, -1].mean()

    def get_attribute_value_loop(self, index):
        unique = np.unique(self.data[:, index])
        return unique

    def get_attribute_index_loop(self):
        return range(self.data.shape[1] - 1)


class RandomTree(Tree):
    def __init__(self, *args, m=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.m = m

    def get_attribute_index_loop(self):
        p = self.data.shape[1] - 1
        if self.m > p:
            raise Exception(f"Size attribute subset greater than number of attributes {p}!")
        return np.random.permutation(p)[:self.m]

    def grow(self):
        depth = self.depth + 1
        # Grow if current depth less than max depth and current data in greater than min group size
        if depth <= self.max_depth and self.data.shape[0] > self.min_group_size:
            # get best split, these are the data for left and right tree
            right, left, condition = self.get_best_split()

            # if split exists, grow tree with increasing depths
            if all(map(lambda x: x is not None, [left, right])):
                self.condition = condition
                self.left = self.__class__(self.max_depth, self.min_group_size, depth, left, m=self.m).grow()
                self.right = self.__class__(self.max_depth, self.min_group_size, depth, right, m=self.m).grow()

        return self


class RandomTreeClassifier(RandomTree, TreeClassifier):
    pass


class RandomTreeRegression(RandomTree, TreeRegression):
    pass
