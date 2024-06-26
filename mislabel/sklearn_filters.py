import copy
from collections import defaultdict
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from mislabel.base import Filter


class ModelFilter(Filter):
    def __init__(self, models: Optional[List] = None, n_folds: int = 5):
        if models is None:
            # Use default ensemble as used in the original paper
            models = [LogisticRegression(), KNeighborsClassifier(), DecisionTreeClassifier()]
        self.models = models
        self.n_folds = n_folds

        self.counts = defaultdict(int)
        self.sums = defaultdict(float)

    def score_samples(self, X: np.array, y: np.array) -> pd.Series:
        # split the dataset into n_folds
        folds = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=42)
        for train_index, test_index in folds.split(X, y):

            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]


            # train the model on the train dataset
            for model in self.models:
                # Create a new instance of the model
                model = copy.deepcopy(model)

                model.fit(X_train, y_train)

                # score the test dataset
                predictions = model.predict(X_test)
                is_correct_a = np.array(predictions == y_test)
                for idx, is_correct in zip(test_index, is_correct_a):
                    self.counts[idx] += 1
                    self.sums[idx] += is_correct

        return self.get_dataframe_score(self.counts, self.sums)


class PCSRandomForest(Filter):
    """Probability-Centric Stratified RandomForest (PCS-RF)"""

    def __init__(self):
        self.counts = defaultdict(int)
        self.sums = defaultdict(float)

    def score_samples(self, X, y) -> pd.Series:
        for _ in range(20):
            # split the dataset into n_folds
            folds = StratifiedKFold(n_splits=2, shuffle=True)
            for train_index, test_index in folds.split(X, y):

                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]

                # train the model on the train dataset
                model = RandomForestClassifier()
                model.fit(X_train, y_train)

                # Get probabilities of the true class
                proba = model.predict_proba(X_test)
                cls = list(model.classes_)
                for idx, p, y_sample in zip(test_index, proba, y_test):
                    self.counts[idx] += 1
                    self.sums[idx] += p[cls.index(y_sample)]

        return self.get_dataframe_score(self.counts, self.sums)


class ExtraTreeFilter(Filter):

    def score_samples(self, X: np.array, y: np.array) -> pd.Series:
        extra_tree = ExtraTreesClassifier(n_estimators=1000, bootstrap=True, max_samples=0.5)

        extra_tree.fit(X,y)
        proba = extra_tree.predict_proba(X)
        cls = list(extra_tree.classes_)
        counts = {idx: 1 for idx in range(len(X))}
        sums = {idx: p[cls.index(y_sample)] for idx, p, y_sample in zip(range(len(X)), proba, y)}

        return self.get_dataframe_score(counts, sums)
