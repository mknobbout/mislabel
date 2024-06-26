from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import auc


def perturbate_y(X, y, fraction=0.1, method='ncar'):
    # Method to perturbate the target variable, in order to detect misclassifications.
    # It should return a new y variable with some perturbations, and also 0/1 mask of the perturbations.
    # The perturbated y should not be the same as the original y.

    # Randomly select a fraction of the target variable to perturbate
    n = int(len(y) * fraction)

    if method == 'ncar':
        # Randomly select n indices
        idx = np.random.choice(len(y), n, replace=False)
    if method == 'nar':
        # Draw a random [0,1] probabilty for each class
        p = {label: np.random.rand() for label in np.unique(y)}
        # Assign each sample a weight based on the probability of the class they have
        weights = np.array([p[label] for label in y])
        # Draw "fraction" samples based on the probability that they have
        idx = np.random.choice(len(y), n, replace=False, p=weights/weights.sum())
    if method == 'nnar':
        X_scaled = (X - X.mean(axis=0)) / X.std(axis=0)
        # Generate random coefficients
        random_coefficients = np.random.normal(size=X_scaled.shape[1])
        # Calculate the random output as a weighted sum of normalized features
        raw_weights = X_scaled.dot(random_coefficients)
        # Apply exponential transformation to get positive weights
        weights = np.exp(raw_weights)
        idx = np.random.choice(len(y), n, replace=False, p=weights/weights.sum())

    y_perturbed = y.copy()

    # All labels
    labels = np.unique(y)

    # Create a mask of the perturbations
    mask = np.zeros(len(y), dtype=bool)



    # Loop through the possible class labels
    for y_class in np.unique(y):
        # Select the indices of the original y where the class is y_class
        idx_class = idx[np.where(y[idx] == y_class)[0]]

        # Choices, it should be labels without y_class
        choices = labels[labels != y_class]

        y_perturbed[idx_class] = np.random.choice(choices, len(idx_class), replace=True)
        mask[idx_class] = True

    return y_perturbed, mask


def score(scores, mask) -> Tuple[pd.DataFrame, float]:
    # Sort the scores by score
    result_df = pd.DataFrame({
        "idx": scores.index,
        "score": scores.values,
        "mislabeled": mask
    }).sort_values(by="score", ascending=False)

    # Calculate the AUC
    return result_df, auc(result_df["score"], 1 - result_df["mislabeled"])

