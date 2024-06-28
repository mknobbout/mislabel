from collections import defaultdict
from typing import Hashable, List, Optional

import numpy as np
import pandas as pd
import torch
import tqdm
from torch.utils.data import Dataset, TensorDataset

from mislabel.base import Scorer


class AUMScorer(Scorer):
    """
    AUMMislabelPredictor is a class that predicts the mislabeling of samples using the AUM method.
    """

    def __init__(
        self,
        hidden_size: Optional[int] = 100,
        batch_size=32,
        epochs=10,
        show_epochs=False,
    ):
        self.batch_size = batch_size
        self.epochs = epochs
        self.hidden_size = hidden_size
        self.show_epochs = show_epochs

    @staticmethod
    def update_scores(
        values: dict,
        counts: dict,
        logits: torch.Tensor,
        targets: torch.Tensor,
        sample_ids: List[Hashable],
    ):
        target_values = logits.gather(1, targets.view(-1, 1)).squeeze()

        # mask out target values
        masked_logits = torch.scatter(logits, 1, targets.view(-1, 1), float("-inf"))
        other_logit_values, _ = masked_logits.max(1)
        other_logit_values = other_logit_values.squeeze()
        margin_values = (target_values - other_logit_values).tolist()

        for sample_id, margin in zip(sample_ids, margin_values):
            values[sample_id] += margin
            counts[sample_id] += 1

    @staticmethod
    def convert_to_dataset(X, y) -> Dataset:

        # Check if y is a numpy array and of type np.int64
        if y.dtype != np.int64:
            # Convert target label to int64 (required by torch)
            target_labels = np.unique(y)
            # Map the target labels to integers
            y = np.array(
                [np.where(target_labels == label)[0][0] for label in y], dtype=np.int64
            )

        return TensorDataset(
            torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.int64)
        )

    def score_samples(self, X: np.array, y: np.array) -> pd.Series:
        dataset = self.__class__.convert_to_dataset(X, y)

        # Since we are only interested in classification, we will use CrossEntropyLoss
        loss_fn = torch.nn.CrossEntropyLoss()

        # Create a new model
        if self.hidden_size is None:
            model = torch.nn.Sequential(torch.nn.Linear(X.shape[1], len(np.unique(y))))
        else:
            model = torch.nn.Sequential(
                torch.nn.Linear(X.shape[1], self.hidden_size),
                torch.nn.ReLU(),
                torch.nn.Linear(self.hidden_size, len(np.unique(y))),
            )

        # Simple Adam optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Create a data loader
        indexed_dataset = IndexedDataset(dataset)
        loader = torch.utils.data.DataLoader(
            indexed_dataset, batch_size=self.batch_size, shuffle=True
        )

        # Create an AUM object
        values = defaultdict(float)
        counts = defaultdict(int)

        # Show a progress bar if show_epochs is True
        if self.show_epochs:
            epochs = tqdm.tqdm(range(self.epochs), desc="Epochs")
        else:
            epochs = range(self.epochs)

        for _ in epochs:
            for batch in loader:
                optimizer.zero_grad()
                (x, y), idx = batch
                y_pred = model(x)
                loss = loss_fn(y_pred, y)

                loss.backward()
                optimizer.step()

                self.__class__.update_scores(values, counts, y_pred, y, idx.numpy())

        return self.get_dataframe_score(values=values, counts=counts)


class IndexedDataset(Dataset):
    """
    IndexedDataset is a wrapper around a torch Dataset that returns the index of the sample along with the sample.
    """

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], index
