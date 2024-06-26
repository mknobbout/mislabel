import torch
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler

from mislabel import (AUMMislabelPredictor, ModelFilter, PCSRandomForest, ExtraTreeFilter,
                      perturbate_y, score)

# Load the iris dataset
data = fetch_openml(name="iris", version=1)
X, y = data["data"], data["target"]
X = StandardScaler().fit_transform(X)

y_prime, mask = perturbate_y(X, y, fraction=0.2)

# Use logistic, knn and tree
#models = [LogisticRegression(), KNeighborsClassifier(), DecisionTreeClassifier()]

scores = ExtraTreeFilter().score_samples(X, y_prime)

_, r = score(scores, mask)
print(r)

scores = AUMMislabelPredictor(
    model=torch.nn.Sequential(
        torch.nn.Linear(4, 10),
        torch.nn.ReLU(),
        torch.nn.Linear(10, 3)
    ),
    batch_size=4,
    epochs=2
).score_samples(X, y_prime)

_, r1 = score(scores, mask)

print('Test 1:', r1)
scores2 = PCSRandomForest().score_samples(X, y_prime)

_, r2 = score(scores2, mask)

print('Test 2:', r2)

scores3 = ModelFilter().score_samples(X, y_prime)

_, r3 = score(scores3, mask)

print('Test 3:', r3)