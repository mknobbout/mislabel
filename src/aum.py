from collections import defaultdict
from typing import List, Hashable
import pandas as pd
import torch

from torchvision import datasets, transforms
from torch.utils.data import Dataset


class AUMCalculator:
    """
    AUMCalculator is a class that calculates the average uncertainty margin (AUM) for each sample in a dataset.
    """
    def __init__(self):
        self.counts = defaultdict(int)
        self.sums = defaultdict(float)

    def update(
        self, logits: torch.Tensor, targets: torch.Tensor, sample_ids: List[Hashable]
    ) -> None:
        """
        Update the AUM calculator with new logits, targets, and sample_ids.
        :param logits: logits from the model
        :param targets: target values
        :param sample_ids: sample ids
        :return: None
        """

        target_values = logits.gather(1, targets.view(-1, 1)).squeeze()

        # mask out target values
        masked_logits = torch.scatter(logits, 1, targets.view(-1, 1), float("-inf"))
        other_logit_values, _ = masked_logits.max(1)
        other_logit_values = other_logit_values.squeeze()
        margin_values = (target_values - other_logit_values).tolist()

        for sample_id, margin in zip(sample_ids, margin_values):
            self.counts[sample_id] += 1
            self.sums[sample_id] += margin

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert the AUM calculator to a pandas DataFrame.
        :return: Pandas DataFrame
        """
        results = [
            {
                "sample_id": sample_id.numpy(),
                "aum": self.sums[sample_id] / self.counts[sample_id],
            }
            for sample_id in self.counts.keys()
        ]

        return pd.DataFrame(results)


class AUMMislabelPredictor:
    """
    AUMMislabelPredictor is a class that predicts the mislabeling of samples using the AUM method.
    """

    def __init__(self, batch_size=32, epochs=10):
        self.batch_size = batch_size
        self.epochs = epochs

    def score_samples(self, model: torch.nn.Module, dataset: Dataset) -> pd.Series:
        """
        Score the samples in the dataset using the AUM method.
        :param model: Model to use for scoring
        :param dataset: Dataset to score
        :return: A pandas Series with the AUM scores. The index of the series should match the index of the dataset.
        """
        # Since we are only interested in classification, we will use CrossEntropyLoss
        loss_fn = torch.nn.CrossEntropyLoss()

        # Simple Adam optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Create a data loader
        indexed_dataset = IndexedDataset(dataset)
        loader = torch.utils.data.DataLoader(
            indexed_dataset, batch_size=self.batch_size, shuffle=True
        )

        # Create an AUM object
        aum = AUMCalculator()

        for _ in range(self.epochs):
            for batch in loader:
                optimizer.zero_grad()
                (x, y), idx = batch
                y_pred = model(x)
                loss = loss_fn(y_pred, y)

                loss.backward()
                optimizer.step()

                aum.update(y_pred, y, idx)

        return (
            aum.to_dataframe().sort_values(by="sample_id").set_index("sample_id")["aum"]
        )


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


# Create a basic torch deep learning model for handwritten images. Also import the data

# Load the data
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))]
)


r = AUMMislabelPredictor().score_samples(
    model=torch.nn.Sequential(
        torch.nn.Linear(784, 128), torch.nn.ReLU(), torch.nn.Linear(128, 10)
    ),
    dataset=datasets.MNIST("data", train=True, download=True, transform=transform),
)
print(r)
