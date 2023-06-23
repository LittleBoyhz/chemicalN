import numpy as np
from scipy.spatial import distance

from .base_method import SamplingMethod


class KCenter(SamplingMethod):
    """This is an implementation of the algorithm proposed in ICLR 18's paper "Active Learning for Convolutional Neural
    Networks: A Core-set Approach".

    Attributes:
        self.active_dataset: (ActiveDataset) An instance of class ActiveDataset.

        self.query(): (function) Uniformly select data points.
        self._grading(): (function) Gives each data points the same score.
    """
    def __init__(self, active_dataset, *args, **kwargs):
        """Initializes a KCenter instance.

        Args:
            active_dataset: (ActiveDataset) An instance of class ActiveDataset.
            args: (tuple) Additional arguments.
            kwargs: (dict) Additional arguments.
        """
        super().__init__(active_dataset, *args, **kwargs)

    def query(self, budget, *args, **kwargs):
        """Select data points.

        Args:
            budget: (int) Number of data to select.
            args: (tuple) Additional arguments.
            kwargs: (dict) Additional arguments.

        Returns:
            indices: (numpy.ndarray) Indices of selected unlabeled data. Shape: (number,).
            is_global: (bool) Whether the returned indices are global or not.
        """
        local_indices = np.zeros((budget,), dtype=np.int64)

        labeled_idx = self.active_dataset.index["labeled"]
        unlabeled_idx = self.active_dataset.index["unlabeled"]

        feature_labeled = self.active_dataset.feature_orig_x_train[labeled_idx, ...]
        feature_unlabeled = self.active_dataset.feature_orig_x_train[unlabeled_idx, ...]

        dist_mat = distance.cdist(feature_unlabeled, feature_labeled)
        dist_mat = np.amin(dist_mat, axis=1, keepdims=True)

        num_selected = 0
        while budget > 0:
            farthest_idx = np.argmax(dist_mat, axis=0)
            local_indices[num_selected] = farthest_idx.item()

            temp_dist_mat = distance.cdist(feature_unlabeled, feature_unlabeled[farthest_idx, ...])
            dist_mat = np.minimum(dist_mat, temp_dist_mat)

            budget -= 1
            num_selected += 1

        return local_indices, False

    def _grading(self, *args, **kwargs):
        pass
