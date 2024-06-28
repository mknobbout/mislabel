import copy
from collections import defaultdict
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from mislabel.base import Scorer


class ModelScorer(Scorer):
    def __init__(self, models: Optional[List] = None, n_folds: int = 5):
        if models is None:
            # Use default ensemble as used in the original paper
            models = [
                # Two logistic regression classifiers
                LogisticRegression(C=1e-2),
                LogisticRegression(C=1.0),
                # Try three different KNN classifiers
                KNeighborsClassifier(n_neighbors=1),
                KNeighborsClassifier(n_neighbors=3),
                KNeighborsClassifier(n_neighbors=5),
                # Try two decision tree classifiers with typical good hyperparameters
                DecisionTreeClassifier(max_depth=5),
                DecisionTreeClassifier(max_depth=None),
            ]
        self.models = models
        self.n_folds = n_folds

    def score_samples(self, X: np.array, y: np.array) -> pd.Series:
        values, counts = defaultdict(float), defaultdict(int)

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
                    counts[idx] += 1
                    values[idx] += is_correct

        return self.get_dataframe_score(values=values, counts=counts)


class RandomForestScorer(Scorer):
    def score_samples(self, X, y) -> pd.Series:
        model = RandomForestClassifier(n_estimators=1000, oob_score=True)
        model.fit(X, y)

        # Get oob score for each sample
        oob_score = model.oob_decision_function_

        # Get probabilities of the true class
        cls = list(model.classes_)
        result = {
            idx: p[cls.index(y_sample)]
            for idx, p, y_sample in zip(range(len(X)), oob_score, y)
        }
        return self.get_dataframe_score(result)
