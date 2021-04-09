import torch
from query_strats.DataManager import DataManager
from .QueryStrategy import QueryStrategy
from models.CNNBaseModel import CNNBaseModel


class EntropyQueryStrategy(QueryStrategy):
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
        self.kwargs = kwargs

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

    def execute_query(self, query_size, model: CNNBaseModel, device='cpu', batch=1):
        """
        This function returns the indices of new data sample to label.

        :param query_size: number of sample to select for labelling
        :param model: model used for predicting scores on which querying is based
        (useless here, used for uniform functions)
        :param device: device to use 'cpu' or 'gpu'
        :param batch: size of sample to evaluate per iteration to use.
        :return:
        idx: indices of unlabeled data sample to label
        """

        unlabeled_data_loader, idx = self.dm.get_unlabelled_data_loader(batch)

        # unlabeled_data, idx = self.dm.get_unlabeled_data()
        # unlabeled_data = unlabeled_data[:]
        entropies = list()
        with torch.no_grad():
            for data in unlabeled_data_loader:
                inputs = data[0].to(device)
                probs = model.predict_proba(inputs)
                log_probs = torch.log(probs)
                entropy = - probs * log_probs
                entropy = torch.sum(entropy, dim=1)
                entropies.append(entropy)

        entropies = torch.cat(entropies)
        entropies_argsort = torch.argsort(entropies, descending=True)

        selected_idx = entropies_argsort[:query_size]

        return idx[selected_idx]
