import torch
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler

from mislabel import (AUMScorer, ModelScorer, RandomForestScorer,
                      perturbate_y, score)

# Load the iris dataset
data = fetch_openml(name="iris", version=1)
X, y = data["data"], data["target"]
X = StandardScaler().fit_transform(X)

y_prime, mask = perturbate_y(X, y, fraction=0.4)

scores = AUMScorer(
    model=torch.nn.Sequential(
        torch.nn.Linear(4, 10),
        torch.nn.ReLU(),
        torch.nn.Linear(10, 3)
    ),
    batch_size=4,
    epochs=5
).score_samples(X, y_prime)
_, r1 = score(scores, mask)
print('AUM:', r1)

scores2 = RandomForestScorer().score_samples(X, y_prime)
_, r2 = score(scores2, mask)
print('RandomForestScorer:', r2)

scores3 = ModelScorer().score_samples(X, y_prime)
_, r3 = score(scores3, mask)
print('ModelScorer:', r3)