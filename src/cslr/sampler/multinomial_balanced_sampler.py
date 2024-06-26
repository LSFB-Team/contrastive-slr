import numpy as np
import torch
from torch.utils.data import Sampler


def _count_labels(labels):
    counts = {}
    instances_labels = np.array([label for label in labels])
    unique_labels = np.unique(instances_labels)
    for label in unique_labels:
        counts[label] = np.sum(instances_labels == label)
    return counts


class MultinomialBalancedSampler(Sampler):
    """
    Randomly samples instances in a balanced way using a multinomial distribution.

    Notes:
        - The weight of an instance is the inverse of the frequency of its label.
        - It is recommended to allow duplicates when over-sampling. If you don't want to allow duplicates and want to
        ensure balanced sampling, you have to perform under-sampling with a number of samples less or equal than the
        minimum label frequency.

    Args:
        labels (list): all the labels in the dataset.
        indices (list, optional): a list of indices. Default=None.
        num_samples (int, optional): number of samples to draw. Default=None.
        allow_duplicates (bool): whether to allow duplicates in the sampling, or not. Default=True.
    """

    def __init__(
        self, labels, *, indices=None, num_samples=None, allow_duplicates=True
    ):
        super().__init__()
        self.indices = list(range(len(labels))) if indices is None else indices
        self.num_samples = len(self.indices) if num_samples is None else num_samples
        self.allow_duplicates = allow_duplicates

        if not allow_duplicates:
            assert num_samples <= len(
                self.indices
            ), "If you don't allow duplicates, the number of samples must be less or equal than the number of indices."

        counts = _count_labels(labels)
        instances_weights = [1.0 / (counts[labels[idx]]) for idx in self.indices]
        self.instances_weights = torch.tensor(instances_weights, dtype=torch.double)

    def __iter__(self):
        return (
            self.indices[i]
            for i in torch.multinomial(
                self.instances_weights,
                self.num_samples,
                replacement=self.allow_duplicates,
            )
        )

    def __len__(self):
        return self.num_samples
