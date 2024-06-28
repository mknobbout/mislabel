import numpy as np
import pandas as pd


class Scorer:

    def score_samples(self, X: np.array, y: np.array) -> pd.Series:
        raise NotImplementedError("score_samples method is not implemented")

    def get_dataframe_score(self, values: dict, counts: dict = None) -> pd.Series:
        """
        Standardize the output of the filter to a pandas DataFrame
        :param counts: Dictionary of counts
        :param sums: Dictionary of sums
        :return: A series with the scores
        """
        results = [
            {
                "idx": sample_id,
                "score": values[sample_id]
                / (counts[sample_id] if counts is not None else 1),
            }
            for sample_id in values.keys()
        ]
        # Compute scores
        scores = pd.DataFrame(results).sort_values(by="idx").set_index("idx")["score"]
        if scores.min() != scores.max():
            # Normalize scores
            scores = (scores - scores.min()) / (scores.max() - scores.min())

        return scores
