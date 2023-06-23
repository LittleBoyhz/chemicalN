import numpy as np

from .base_method import SamplingMethod


class UniformSampling(SamplingMethod):
    """The uniform sampling algorithm.

    Attributes:
        self.active_dataset: (ActiveDataset) An instance of class ActiveDataset.

        self.query(): (function) Uniformly selects data points.
        self._grading(): (function) Gives each data points the same score.
    """
    def __init__(self, active_dataset, *args, **kwargs):
        """Initializes a UniformSampling instance.

        Args:
            active_dataset: (ActiveDataset) An instance of class ActiveDataset.
            args: (tuple) Additional arguments.
            kwargs: (dict) Additional arguments.
        """
        super().__init__(active_dataset, *args, **kwargs)

    def query(self, budget, *args, **kwargs):
        """Uniformly selects data points.

        Args:
            budget: (int or float) Number of data to select. Float values indicate a percentage.
            args: (tuple) Additional arguments.
            kwargs: (dict) Additional arguments.

        Returns:
            indices: (numpy.ndarray) Indices of selected unlabeled data. Shape: (number,).
            is_global: (bool) Whether the returned indices are global or not.
        """
        if budget < 1.0:  # convert to int
            budget = round(budget * self.active_dataset.orig_x_train.shape[0])
        else:
            budget = round(budget)

        if self.active_dataset.num_unlabeled <= budget:
            return self.active_dataset.index["unlabeled"], True

        local_indices = np.random.choice(self.active_dataset.num_unlabeled, budget, replace=False)
        return local_indices, False

    def _grading(self, *args, **kwargs):
        """Gives each data point the same score.

        Args:
            args: (tuple) Additional arguments.
            kwargs: (dict) Additional arguments.

        Returns:
            scores: (numpy.ndarray) Scores of unlabeled data. Shape: (number,).
        """
        scores = np.ones((self.active_dataset.num_unlabeled,))
        return scores
