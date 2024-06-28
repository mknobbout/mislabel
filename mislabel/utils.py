import itertools
import os
from typing import Tuple

import numpy as np
import pandas as pd
import tqdm
from sklearn.metrics import auc
from sklearn.model_selection import train_test_split

from mislabel import AUMScorer, ModelScorer, RandomForestScorer


def perturbate_y(X, y, fraction=0.1, method="ncar") -> Tuple[np.array, np.array]:
    """
    Method to perturbate the target variable, in order to detect misclassifications.
    It should return a new y variable with some perturbations, and also 0/1 mask of the perturbations.
    The perturbated y should not be the same as the original y. Whenever the mask
    of a data point is 1, the corresponding y should be different from the original y, it is
    treated as a mislabel in the evaluation.
    :param X: The features
    :param y: The target variable
    :param fraction: The fraction of the target variable to perturbate
    :param method: Method to perturbate the target variable (ncar, nar, nnar)
    :return: Tuple of the perturbated y and the mask of the perturbations
    """

    # Randomly select a fraction of the target variable to perturbate
    n = int(len(y) * fraction)

    if method == "ncar":
        # Randomly select n indices
        idx = np.random.choice(len(y), n, replace=False)
    if method == "nar":
        # Draw a random [0,1] probabilty for each class
        p = {label: np.random.rand() for label in np.unique(y)}
        # Assign each sample a weight based on the probability of the class they have
        weights = np.array([p[label] for label in y])
        # Draw "fraction" samples based on the probability that they have
        idx = np.random.choice(len(y), n, replace=False, p=weights / weights.sum())
    if method == "nnar":
        # Select a random subset of 10 columns of X
        X_r = X[:, np.random.choice(X.shape[1], min(X.shape[1], 10), replace=False)]
        # Normalize the features
        X_scaled = (X_r - X_r.mean(axis=0)) / X_r.std(axis=0)
        # Generate random coefficients
        random_coefficients = np.random.normal(size=X_scaled.shape[1])
        # Calculate the random output as a weighted sum of normalized features
        raw_weights = X_scaled.dot(random_coefficients)
        # Apply exponential transformation to get positive weights
        weights = np.exp(raw_weights)
        # Normalize the weights
        idx = np.random.choice(len(y), n, replace=False, p=weights / weights.sum())

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


def get_data(experiment) -> Tuple[np.array, np.array]:
    """
    Load the data for the experiment
    :param experiment: Name of the experiment
    :return: Tuple of X and y
    """

    if experiment not in ["iris", "newsgroups", "covtype"]:
        raise ValueError("Invalid experiment")

    # Load the data for the experiment
    if experiment == "iris":
        from sklearn.datasets import fetch_openml

        data = fetch_openml(name="iris", version=1)
        X, y = data["data"], data["target"]
        X = (X - X.mean(axis=0)) / X.std(axis=0)
        X = X.values
    if experiment == "newsgroups":
        from sklearn.datasets import fetch_20newsgroups_vectorized
        from sklearn.decomposition import TruncatedSVD

        data = fetch_20newsgroups_vectorized(subset="test")
        X, y = data["data"], data["target"]
        # SVD to 1024 components
        X = TruncatedSVD(n_components=1024).fit_transform(X)
    if experiment == "covtype":
        from sklearn.datasets import fetch_covtype

        data = fetch_covtype()
        X, y = data["data"], data["target"]
        # Select 20% from X,y
        X, _, y, _ = train_test_split(X, y, test_size=0.9, stratify=y, random_state=42)
        X = (X - X.mean(axis=0)) / X.std(axis=0)

    return X, y


def run_experiment(experiment, n_runs=10) -> pd.DataFrame:
    """
    Run the experiment
    :param experiment: Name of the experiment
    :param n_runs: Number of runs
    :return: A dataframe with the results
    """
    # Get absolute path of this file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Create a directory to store the results
    results_dir = os.path.join(current_dir, "..", "results")
    os.makedirs(results_dir, exist_ok=True)
    # Experiment csv
    experiment_csv = os.path.join(results_dir, f"{experiment}.csv")

    # Load the data
    X, y = get_data(experiment)

    if os.path.isfile(experiment_csv):
        print(f"Loading previous results for {experiment}")
        result = pd.read_csv(experiment_csv).to_dict(orient="records")
    else:
        print(f"Running experiment for {experiment}")
        result = []

    methods = ["ncar", "nar", "nnar"]
    fractions = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    runs = range(n_runs)
    algorithms = ["AUMScorer", "RandomForestScorer", "ModelScorer"]

    def run_algorithm(X, y, mask, method, fraction, run, algorithm):
        """
        Run the algorithm
        :param X: Features
        :param y: Target variable
        :param mask: Mask of the mislabelings
        :param method: Method to perturbate the target variable
        :param fraction: Fraction of the target variable to perturbate
        :param run: Run number
        :param algorithm: Algorithm to use
        """
        if algorithm == "AUMScorer":
            params = {
                "iris": {"hidden_size": None, "batch_size": 32, "epochs": 12},
                "newsgroups": {"hidden_size": 100, "batch_size": 32, "epochs": 150},
                "covtype": {"hidden_size": 100, "batch_size": 32, "epochs": 150},
            }
            model = AUMScorer(**params[experiment])
        elif algorithm == "RandomForestScorer":
            model = RandomForestScorer()
        elif algorithm == "ModelScorer":
            model = ModelScorer()
        else:
            raise ValueError("Invalid algorithm")

        scores = model.score_samples(X, y)
        _, auc_score = auc_score_mask(scores, mask)

        return {
            "method": method,
            "fraction": fraction,
            "run": run,
            "algorithm": algorithm,
            "auc_score": auc_score,
        }

    all_runs = set(itertools.product(methods, fractions, runs, algorithms))
    runs_done = set(
        [(x["method"], x["fraction"], x["run"], x["algorithm"]) for x in result]
    )
    runs_to_do = all_runs - runs_done
    print(f"Runs to do: {len(runs_to_do)} (out of {len(all_runs)})")

    # Compute cartesian product
    for method, fraction, run, algorithm in tqdm.tqdm(runs_to_do):
        # Perturbate the target variable
        y_prime, mask = perturbate_y(X, y, method=method, fraction=fraction)

        # Run the experiments
        result.append(run_algorithm(X, y_prime, mask, method, fraction, run, algorithm))

        pd.DataFrame(result).to_csv(experiment_csv, index=False)

    return pd.DataFrame(result)


def auc_score_mask(scores, mask) -> Tuple[pd.DataFrame, float]:
    """
    Calculate the AUC score of the scores with the mask. The mask is a boolean array
    representing the mislabelings. The scores are the probabilities/rankings of the samples, where
    a lower score means that the sample is more likely to be mislabeled.
    :param scores: The scores of the samples
    :param mask: The original mask of the samples
    :return: Scores and the AUC score
    """
    # Sort the scores by score
    # A higher score means that the sample is more likely to be correctly labeled.
    result_df = pd.DataFrame({"score": scores.values, "mislabeled": mask}).sort_values(
        by="score", ascending=False
    )

    # Calculate the AUC
    return result_df, auc(result_df["score"], 1 - result_df["mislabeled"])
