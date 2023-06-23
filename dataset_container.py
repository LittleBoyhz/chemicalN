import os

import numpy as np


def get_dataset(dataset, dataset_dir, representation=None, representation_dim=None, split_mode=None, pred=True):
    """Fetches dataset from disk.

    Supported datasets are {"gaussian_1", "gaussian_2"}.

    Args:
        dataset: (str) Name of the dataset. Supported dataset names are {"gaussian_1", "gaussian_2"}.
        dataset_dir: (str) The directory where dataset is stored.
        representation: (str) The method for representing a molecule.
        representation_dim: (int) The dimension of the representation of a molecule.
        pred: (bool) Whether or not to return data that will be used for prediction.

    Returns:
        x_train: (numpy.ndarray) Training data of shape (n_samples, n_features).
        y_train: (numpy.ndarray) Training labels of shape (n_samples,).
        x_test: (numpy.ndarray) Test data of shape (n_samples, n_features).
        y_test: (numpy.ndarray) Test labels of shape (n_samples,).
    """
    if representation is not None:
        dataset_file_predix = os.path.join(dataset_dir, dataset)
        if split_mode is not None:
            dataset_file_predix = os.path.join(dataset_file_predix, "split_" + str(split_mode))
        dataset_file_predix = os.path.join(dataset_file_predix, dataset + "_" + representation,
                                           dataset + "_" + representation if representation_dim is None else
                                           dataset + "_" + representation + "_" + str(representation_dim))
    else:
        dataset_file_predix = os.path.join(dataset_dir, dataset)
        if split_mode is not None:
            dataset_file_predix = os.path.join(dataset_file_predix, "split_" + str(split_mode))
        dataset_file_predix = os.path.join(dataset_file_predix, dataset)
    path_train_data = dataset_file_predix + "_train.npz"
    path_test_data = dataset_file_predix + "_test.npz"
    if pred:
        path_pred_data = dataset_file_predix + "_pred.npz"

    data_train = np.load(path_train_data)
    data_test = np.load(path_test_data)
    if pred:
        data_pred = np.load(path_pred_data)

    x_train, y_train = data_train["train_data"], data_train["train_labels"]
    x_test, y_test = data_test["test_data"], data_test["test_labels"]
    if pred:
        x_pred, y_pred = data_pred["pred_data"], data_pred["pred_labels"]

    if pred:
        return (x_train, y_train), (x_test, y_test), (x_pred, y_pred)
    else:
        return (x_train, y_train), (x_test, y_test)


class ActiveDataset:
    """A dataset that stores all data and indices of labeled and unlabeled data.

    Attributes:
        self.index: (dict) Indices of labeled and unlabeled data.
            self.index["labeled"]: (numpy.ndarray) Indices of labeled data. Shape: (n_samples,).
            self.index["unlabeled"]: (numpy.ndarray) Indices of unlabeled data. Shape: (n_samples,).
        self.orig_x_train: (numpy.ndarray) Original training data. Shape: (n_samples, n_features).
        self.orig_y_train: (numpy.ndarray) Labels of original training data. Shape: (n_samples,).

        self.move_from_unlabeled_to_labeled(): (function) Moves the newly selected unlabeled data to labeled data.
        self.num_labeled(): (function, @property) Returns the number of labeled data.
        self.num_unlabeled(): (function, @property) Returns the number of unlabeled data.
        self._set_init_index(): (function) Splits original training data into initial labeled and unlabeled data.
    """
    def __init__(self, orig_x_train, orig_y_train, init_num_labeled, existing_labeled_idx=None):
        """Initializes an ActiveDataset instance.

        Args:
            orig_x_train: (numpy.ndarray) Original training data. Shape: (n_samples, n_features).
            orig_y_train: (numpy.ndarray) Labels of original training data. Shape: (n_samples,).
            init_num_labeled: (int or float) Number of initial labeled data. Float values indicate a percentage.
        """
        self.orig_x_train = orig_x_train
        self.orig_y_train = orig_y_train
        self.index = dict()
        self.index["labeled"], self.index["unlabeled"], self.num_existing_labeled = \
            self._set_init_index(init_num_labeled, existing_labeled_idx)

    @property
    def num_labeled(self):
        """Returns the number of labeled data.
        """
        return self.index["labeled"].shape[0]

    @property
    def num_unlabeled(self):
        """Returns the number of unlabeled data.
        """
        return self.index["unlabeled"].shape[0]

    def move_from_unlabeled_to_labeled(self, indices, is_global):
        """Moves the newly selected unlabeled data to labeled data.

        Args:
            indices: (numpy.ndarray) Indices of unlabeled data. Shape: (number,).
            is_global: (bool) Whether or not indices is global.
        """
        if is_global:
            updated_labeled_idx = np.concatenate((self.index["labeled"], indices), axis=0)
            updated_unlabeled_idx = np.setdiff1d(self.index["unlabeled"], indices)
        else:
            new_labeled_idx = self.index["unlabeled"][indices, ...]
            updated_labeled_idx = np.concatenate((self.index["labeled"], new_labeled_idx), axis=0)
            updated_unlabeled_idx = np.delete(self.index["unlabeled"], indices, axis=0)

        self.index["labeled"] = updated_labeled_idx
        self.index["unlabeled"] = updated_unlabeled_idx

    def uniformly_convert_unlabeled_to_labeled(self, num_new_labeled):
        shuffled_index = np.random.permutation(self.num_unlabeled)
        self.move_from_unlabeled_to_labeled(shuffled_index[:min(num_new_labeled, self.num_unlabeled)], False)

    def _set_init_index(self, init_num_labeled, existing_labeled_idx=None):
        """Splits original training data into initial labeled and unlabeled data.

        Args:
            init_num_labeled: (int or float) Number of initial labeled data. Float values indicate a percentage.

        Returns:
            shuffled_index[:init_num_labeled]: (numpy.ndarray) Indices of initial labeled data. Shape: (number,).
            shuffled_index[init_num_labeled:]: (numpy.ndarray) Indices of initial unlabeled data. Shape: (number,).
        """
        if existing_labeled_idx is not None:
            num_existing_labeled = existing_labeled_idx.shape[0]
            idx_unlabeled = np.arange(self.orig_x_train.shape[0])[num_existing_labeled:]
            shuffled_index = np.random.permutation(idx_unlabeled.shape[0])

            idx_new_labeled = idx_unlabeled[shuffled_index[:init_num_labeled]]
            idx_unlabeled = idx_unlabeled[shuffled_index[init_num_labeled:]]

            idx_labeled = np.concatenate((existing_labeled_idx, idx_new_labeled), axis=0)

            return idx_labeled, idx_unlabeled, num_existing_labeled

        if init_num_labeled == -1:
            init_num_labeled = self.orig_x_train.shape[0]
        elif init_num_labeled < 1.0:
            init_num_labeled = round(init_num_labeled * self.orig_x_train.shape[0])
        else:
            init_num_labeled = round(init_num_labeled)

        shuffled_index = np.random.permutation(self.orig_x_train.shape[0])
        return shuffled_index[:init_num_labeled], shuffled_index[init_num_labeled:], 0
