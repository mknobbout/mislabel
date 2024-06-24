from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import KFold, StratifiedKFold
from typing import List, Hashable
from collections import defaultdict
import pandas as pd
import copy

class ModelFilter:
    def __init__(self, models: List, n_folds: int):
        self.models = models
        self.n_folds = n_folds

        self.counts = defaultdict(int)
        self.sums = defaultdict(float)

    def score_samples(self, X, y) -> pd.Series:
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
                eval = predictions == y_test
                for idx, is_correct in zip(test_index, eval):
                    self.counts[idx] += 1
                    self.sums[idx] += is_correct

        results = [
            {
                "idx": sample_id,
                "score": self.sums[sample_id] / self.counts[sample_id],
            }
            for sample_id in self.counts.keys()
        ]

        return pd.DataFrame(results).sort_values(by="idx").set_index("idx")["score"]



