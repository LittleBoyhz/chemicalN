import abc
import copy


class SamplingMethod(metaclass=abc.ABCMeta):
    """An abstract base class for implementing active learning algorithms.

    Attributes:
        self.active_dataset: (ActiveDataset) An instance of class ActiveDataset.

        self.query(): (function) Returns the local or global indices of selected data.
        self._grading(): (function) Grades each data points and returns the scores.
    """
    def __init__(self, active_dataset, deep_copy=False, *args, **kwargs):
        """Initializes a SamplingMethod instance.

        Args:
            active_dataset: (ActiveDataset) An instance of class ActiveDataset.
            deep_copy: (bool) If True, then active_dataset will be deeply copied.
            args: (tuple) Additional arguments.
            kwargs: (dict) Additional arguments.
        """
        if deep_copy:
            self.active_dataset = copy.deepcopy(active_dataset)
        else:
            self.active_dataset = active_dataset

    @abc.abstractmethod
    def query(self, budget, *args, **kwargs):
        """Returns the indices of selected unlabeled data.

        Args:
            budget: (int or float) Number of data to select. Float values indicate a percentage.
            args: (tuple) Additional arguments.
            kwargs: (dict) Additional arguments.
        """
        pass

    @abc.abstractmethod
    def _grading(self, *args, **kwargs):
        """Grades each data point and returns the scores.

        Args:
            args: (tuple) Additional arguments.
            kwargs: (dict) Additional arguments.
        """
        pass
