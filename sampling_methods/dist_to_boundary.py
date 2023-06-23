import numpy as np

from .base_method import SamplingMethod


class DistToBoundary(SamplingMethod):
    """An active learning algorithm that selects data points which are closest to the decision boundary.

        Attributes:
            self.active_dataset: (ActiveDataset) An instance of class ActiveDataset.

            self.query(): (function) Selects data points that are closest to the decision boundary.
            self._grading(): (function) Calculates and returns the distance of unlabeled data points to the decision
                boundary.
    """
    def __init__(self, active_dataset, *args, **kwargs):
        """Initializes a DistToBoundary instance.

        Args:
            active_dataset: (ActiveDataset) An instance of class ActiveDataset.
            args: (tuple) Additional arguments.
            kwargs: (dict) Additional arguments.
        """
        super().__init__(active_dataset, *args, **kwargs)

    def query(self, budget, *args, **kwargs):
        """Selects data points that are closest to the decision boundary.

        Args:
            budget: (int or float) Number of data to select. Float values indicate a percentage.
            args: (tuple) Additional arguments.
            kwargs: (dict) Additional arguments.

        Returns:
            indices: (numpy.ndarray) Indices of selected unlabeled data. Shape: (n_samples,).
            is_global: (bool) Whether or not the returned indices are global.
        """
        if budget < 1.0:  # convert to int
            budget = round(budget * self.active_dataset.orig_x_train.shape[0])
        else:
            budget = round(budget)

        if self.active_dataset.num_unlabeled <= budget:
            return self.active_dataset.index["unlabeled"], True

        dist_to_boundary = self._grading(*args, **kwargs)  # calculate the distance to decision boundary
        # select the data points that are closest to decision boundary
        local_indices = np.argpartition(dist_to_boundary, budget)[0:budget]
        return local_indices, False

    def _grading(self, model, preprocessor, *args, **kwargs):
        """Calculates and returns the distance of unlabeled data points to the decision boundary.

        Args:
            model: The model that is used to compute the distance to decision boundary.
            preprocessor: Tools for data pre-processing.
            args: (tuple) Additional arguments.
            kwargs: (dict) Additional arguments.

        Returns:
            dist_to_boundary: (numpy.ndarray) Distances to decision boundary. Shape: (n_samples,).
        """
        unlabeled_idx = self.active_dataset.index["unlabeled"]
        x = preprocessor.transform(self.active_dataset.orig_x_train[unlabeled_idx, ...])
        dist_to_boundary = np.abs(model.decision_function(x))
        return dist_to_boundary
