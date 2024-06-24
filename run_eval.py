# Load credit approval dataset
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from src.filter import ModelFilter
from sklearn.metrics import auc
import pandas as pd
import numpy as np

def perturbate_y(X, y, fraction=0.1):
    # Method to perturbate the target variable, in order to detect misclassifications.
    # It should return a new y variable with some perturbations, and also 0/1 mask of the perturbations.
    # The perturbated y should not be the same as the original y.

    # Randomly select a fraction of the target variable to perturbate
    n = int(len(y) * fraction)
    idx = np.random.choice(len(y), n, replace=False)
    y_perturbed = y.copy()

    # All labels
    labels = np.unique(y)

    # Create a mask of the perturbations
    mask = np.zeros(len(y), dtype=bool)

    # Loop through the possible class labels
    for y_class in np.unique(y):
        # Select the indices of the original y where the class is y_class
        idx_class = idx[np.where(y[idx] == y_class)[0]]

        # Choices, it should be np.unique(y) without i
        choices = labels[labels != y_class]

        y_perturbed[idx_class] = np.random.choice(choices, len(idx_class), replace=True)
        mask[idx_class] = True

    return y_perturbed, mask


data = fetch_openml(name="iris", version=1)
X, y = data["data"], data["target"]
X = StandardScaler().fit_transform(X)

y_prime, mask = perturbate_y(X, y, fraction=0.05)

# Use logistic, knn and tree
models = [LogisticRegression(), KNeighborsClassifier(), DecisionTreeClassifier()]

scores = ModelFilter(models=models, n_folds=5).score_samples(X, y_prime)

# Sort the scores by score
result = pd.DataFrame({
    "idx": scores.index,
    "score": scores.values,
    "mislabeled": mask
}).sort_values(by="score", ascending=False)

# Calculate the AUC
print(auc(result["score"], 1-result["mislabeled"]))