# -*- coding:utf-8 -*-

import warnings
import torch
import numpy as np
from DataManager import DataManager as DM
from query_strats.QueryStrategy import QueryStrategy
from typing import Callable, Type
from tqdm import tqdm


class TrainTestManager(object):
    """
    Class used to train and test model given model and query strategy
    """

    def __init__(self, model, querier,
                 trainset: torch.utils.data.Dataset,
                 testset: torch.utils.data.Dataset,
                 loss_fn: torch.nn.Module,
                 optimizer_factory: Callable[[torch.nn.Module], torch.optim.Optimizer],
                 batch_size=1,
                 initial_train_dataset_ratio=0.2,
                 validation=None,
                 use_cuda=False):
        """
        Args:
            model: model to train
            querier: query_strategy object for active learning
            trainset: dataset used to train the model
            testset: dataset used to test the model
            batch_size: size of minibatch
            initial_train_dataset_ratio: percentage of data queried
            for first iteration of active learning process
            loss_fn: the loss function used
            optimizer_factory: A callable to create the optimizer. see optimizer function
            below for more details
            validation: wether to use custom validation data or let the one by default
            use_cuda: to Use the gpu to train the model
        """
        device_name = 'cuda:0' if use_cuda else 'cpu'
        if use_cuda and not torch.cuda.is_available():
            warnings.warn("CUDA is not available. Suppress this warning by passing "
                          "use_cuda=False to {}()."
                          .format(self.__class__.__name__), RuntimeWarning)
            device_name = 'cpu'

        self.device = torch.device(device_name)
        if validation is not None:
            self.data_manager = DM(trainset, testset, batch_size=batch_size, validation=validation,
                                   initial_train_dataset_ratio=initial_train_dataset_ratio)
        else:
            self.data_manager = DM(trainset, testset, batch_size=batch_size,
                                   initial_train_dataset_ratio=initial_train_dataset_ratio)
        self.loss_fn = loss_fn
        self.model = model
        self.querier = querier
        self.optimizer = optimizer_factory(self.model)
        self.model = self.model.to(self.device)
        self.use_cuda = use_cuda
        self.metric_values = {}

    def training_iteration(self, num_epochs):
        """
        Train the model for num_epochs times on given data
        Args:
            num_epochs: number of times to train the model
        """

    def train(self, num_epochs, complete_data_ratio):
        """
        Train the model until reaching complete_data_ratio of labeled instances
        """

    def evaluate_on_validation_set(self):
        """
        function that evaluate the model on the validation set every epoch
        """

    def evaluate_on_test_set(self):
        """
        function that evaluate the model on the test set every iteration of the
        active learning process
        """

    def accuracy(self, outputs, labels):
        """
        Computes the accuracy of the model
        Args:
            outputs: outputs predicted by the model
            labels: real outputs of the data
        Returns:
            Accuracy of the model
        """


def optimizer_setup(optimizer_class: Type[torch.optim.Optimizer], **hyperparameters) -> \
        Callable[[torch.nn.Module], torch.optim.Optimizer]:
    """
    Creates a factory method that can instanciate optimizer_class with the given
    hyperparameters.

    Why this? torch.optim.Optimizer takes the model's parameters as an argument.
    Thus we cannot pass an Optimizer to the CNNBase constructor.

    Args:
        optimizer_class: optimizer used to train the model
        **hyperparameters: hyperparameters for the model
        Returns:
            function to setup the optimizer
    """

    def f(model):
        return optimizer_class(model.parameters(), **hyperparameters)

    return f
