from abc import ABC, abstractmethod

from query_strats.DataManager import DataManager


class QueryStrategy(ABC):
    """
    This class define a pooling strategy for active learning
    """

    def __init__(self, dm: DataManager, **kwargs):
        """
        Initialize the query strategy object
        :param dataset: dataset used in the pooling strategy implemented.
        :param kwargs: others key arguments to use.
        """
        self.dm = dm

    def get_datamanager(self):
        return self.dm

    # def update

    @abstractmethod
    def execute_query(self):
        """
        This function returns the indices of new data sample to label
        :return:
        idx: indices of unlabeled data sample to label
        """
        pass
