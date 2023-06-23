from .base_method import SamplingMethod
from utils import *


def train(x_train, y_train, model, preprocessor, boundary_value):
    x = preprocessor.fit_transform(x_train)
    y = utils.real_to_binary(y_train, boundary_value)
    if np.unique(y).shape[0] == 1:
        return -1
    model.fit(x, y)


class QBC(SamplingMethod):
    """The Query by Committee algorithm.

    Attributes:
        self.active_dataset: (ActiveDataset) An instance of class ActiveDataset.
        self.num_models: (int) The number of committee members.
        self.num_samples: (int or float) The number of labeled data that is used to train a committee member. Float
            values indicate a percentage.

        self.query(): (function) Selects the data points that have the smallest distance to any of the boundary of the
            committee members.
        self._grading(): (function) Gives each data points the same score.
    """
    def __init__(self, active_dataset, num_models, num_samples, strategy, *args, **kwargs):
        """Initializes a QBC instance.

        Args:
            active_dataset: (ActiveDataset) An instance of class ActiveDataset.
            num_models: (int) The number of committee members.
            num_samples: (int or float) The number of labeled data that is used to train a committee member. Float
                values indicate a percentage.
            args: (tuple) Additional arguments.
            kwargs: (dict) Additional arguments.
        """
        super().__init__(active_dataset, *args, **kwargs)
        self.num_models = num_models
        self.num_samples = num_samples
        self.strategy = strategy

    def query(self, budget, *args, **kwargs):
        """Selects the data points that have the smallest distance to any of the boundary of the committee members.

        Args:
            budget: (int or float) Number of data to select. Float values indicate a percentage.
            args: (tuple) Additional arguments.
            kwargs: (dict) Additional arguments.

        Returns:
            indices: (numpy.ndarray) Indices of selected unlabeled data. Shape: (n_samples,).
            is_global: (bool) Whether the returned indices are global or not.
        """
        if budget < 1.0:  # convert to int
            budget = round(budget * self.active_dataset.orig_x_train.shape[0])
        else:
            budget = round(budget)

        if self.active_dataset.num_unlabeled <= budget:
            return self.active_dataset.index["unlabeled"], True

        score = self._grading(*args, **kwargs)
        local_indices = np.argpartition(score, budget)[0:budget]
        return local_indices, False

    def _grading(self, model_class, model_init_args, preprocessor_class, boundary_value, *args, **kwargs):
        """Computes the smallest distance of each unlabeled data to any of the decision boundary of the committee members.

        Args:
            model_class: A class for initializing a model.
            model_init_args: The arguments for model initialization.
            preprocessor_class: A class for initializing data pre-processing tools.
            boundary_value: (float) We'd like to train a model that is able to distinguish between data points with
                values greater or less than boundary_value.
            args: (tuple) Additional arguments.
            kwargs: (dict) Additional arguments.
        """
        if type(self.num_samples) is int:
            num_samples = self.num_samples
        else:
            num_samples = round(self.num_samples * self.active_dataset.num_labeled)

        score = np.zeros((self.num_models, self.active_dataset.num_unlabeled))

        for i in range(self.num_models):
            model = model_class(**model_init_args)
            preprocessor = preprocessor_class()

            successful = -1
            while successful == -1:
                # randomly select a proportion of the labeled data
                local_indices = np.random.choice(self.active_dataset.num_labeled, num_samples, replace=False)
                global_indices = self.active_dataset.index["labeled"][local_indices]
                x_train = self.active_dataset.orig_x_train[global_indices, ...]
                y_train = self.active_dataset.orig_y_train[global_indices, ...]

                successful = train(x_train, y_train, model, preprocessor, boundary_value)

            unlabeled_idx = self.active_dataset.index["unlabeled"]
            x_unlabeled = preprocessor.transform(self.active_dataset.orig_x_train[unlabeled_idx, ...])
            if self.strategy == "dist_to_boundary":
                score[i, :] = np.abs(model.decision_function(x_unlabeled))
            elif self.strategy in ["margin", "least_confidence"]:
                probabilities = model.predict_proba(x_unlabeled)
                if self.strategy == "margin":
                    score[i, :] = np.absolute(probabilities[:, 0] - probabilities[:, 1])
                elif self.strategy == "least_confidence":
                    score[i, :] = probabilities.max(axis=1)

        return np.amin(score, axis=0)
