from abc import ABC, abstractmethod

from prog.src.DataManager import DataManager


class QueryStrategy(ABC):
    """
    This class define a pooling strategy for active learning
    """

    def __init__(self, dataset: DataManager, **kwargs):
        """
        Initialize the query strategy object
        :param dataset: dataset used in the pooling strategy implemented.
        :param kwargs: others key arguments to use.
        """
        self.dataset = dataset

    def get_dataset(self):
        return self.dataset

    # def update

    @abstractmethod
    def execute_query(self):
        """
        This function returns the indices of new data sample to label
        :return:
        idx: indices of unlabeled data sample to label
        """
        pass
