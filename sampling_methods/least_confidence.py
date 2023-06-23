import numpy as np

from .base_method import SamplingMethod


class LeastConfidence(SamplingMethod):
    """This is an implementation of the LeastConfidence Sampling algorithm.

        Attributes:
            self.active_dataset: (ActiveDataset) An instance of class ActiveDataset.

            self.query(): (function) Selects data points based on the margin scores, i.e., the difference between the
                probabilities of the first and second most probable classes.
            self._grading(): (function) Calculates and returns the probability of the most probable class.
        """

    def __init__(self, active_dataset, *args, **kwargs):
        """Initializes a LeastConfidence instance.

        Args:
            active_dataset: (ActiveDataset) An instance of class ActiveDataset.
            args: (tuple) Additional arguments.
            kwargs: (dict) Additional arguments.
        """
        super().__init__(active_dataset, *args, **kwargs)

    def query(self, budget, *args, **kwargs):
        """Selects data points based on the least confidence scores, i.e., the probability of the most probable class.

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

        uncertainty = self._grading(*args, **kwargs)
        # select the data points with the smallest margin scores
        local_indices = np.argpartition(uncertainty, -budget)[-budget:]
        return local_indices, False

    def _grading(self, model, preprocessor, *args, **kwargs):
        """Calculates and returns the probability of the most probable class.

        Args:
            model: The model that is used to compute the classification probabilities.
            preprocessor: Tools for data pre-processing.
            args: (tuple) Additional arguments.
            kwargs: (dict) Additional arguments.

        Returns:
            uncertainty: (numpy.ndarray) Classification uncertainty of unlabeled data. Shape: (n_samples,).
        """
        unlabeled_idx = self.active_dataset.index["unlabeled"]
        x = preprocessor.transform(self.active_dataset.orig_x_train[unlabeled_idx, ...])
        probabilities = model.predict_proba(x)
        uncertainty = 1 - probabilities.max(axis=1)
        return uncertainty
