import torch
from query_strats.DataManager import DataManager
from .QueryStrategy import QueryStrategy


class RandomQueryStrategy(QueryStrategy):
    """
    This class define a pooling strategy for active learning that randomly choose sample to label
    """

    def __init__(self, dm: DataManager, **kwargs):
        """
        Initialize the query strategy object
        :param dataset: dataset used in the pooling strategy implemented.
        :param kwargs: others key arguments to use.
        """
        super().__init__(dm, **kwargs)
        self.__name__ = 'Random'

    def get_datamanager(self):
        return self.dm

    def update_label(self, idx):
        """
        This function updates the train data loader by adding newly labelled sample.
        :param idx: Indices of sample to add.
        :return:
        dm : Updated data manager
        """
        self.dm.update_train_set(idx)
        return self.dm

    def execute_query(self, query_size, model, device='cpu', batch=1):
        """
        This function returns the indices of new data sample to label.

        :param query_size: number of sample to select for labelling
        :param model: model used for predicting scores on which querying is based
        (useless here, used for uniform functions)
        :param device: device to use 'cpu' or 'gpu'
        :return:
        idx: indices of unlabeled data sample to label
        """
        _, idx = self.dm.get_unlabeled_data()
        shuffled_idx = torch.randperm(len(idx)).long()
        selected_idx = shuffled_idx[:query_size]

        return idx[selected_idx]
