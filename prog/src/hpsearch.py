#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
University of Sherbrooke
NN Class Project
Authors: D'Jeff Kanda, Gabriel McCarthy, Mohamed Ragued
"""

import argparse
from itertools import product

import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from TrainTestManager import TrainTestManager, optimizer_setup
from models.DenseNet import DenseNet
from models.ResNet import ResNet
from models.SENet import SENet
from models.SimpleModel import SimpleModel
from query_strats.DataManager import DataManager as DM
from query_strats.EntropyQueryStrategy import EntropyQueryStrategy
from query_strats.LCQueryStrategy import LCQueryStrategy
from query_strats.MSQueryStrategy import MSQueryStrategy
from query_strats.RandomQueryStrategy import RandomQueryStrategy
from utils import get_data, check_dir


def argument_parser():
    """
        A parser to allow user to easily experiment different models along with datasets and differents parameters
    """
    parser = argparse.ArgumentParser(usage='\n python3 hpsearch.py [model] [dataset] [hyper_parameters]'
                                           '\n python3 hpsearch.py --model SENet [hyper_parameters]',
                                     description="This program make a search of hyper parameter for different models "
                                                 "of classification on"
                                                 " different datasets. Active learning is used to prioritise sample "
                                                 "selection in the training process"
                                     )
    parser.add_argument('--model', type=str, default="BasicCNN",
                        choices=["BasicCNN", "SENet", "ResNet", "DenseNet"])
    parser.add_argument('--dataset', type=str, default="mnistfashion", choices=["cifar100", "mnistfashion"])

    parser.add_argument('--validation', type=float, default=0.1,
                        help='Percentage of training data to use for validation')
    parser.add_argument('--query_strategy', type=str, default='Random',
                        choices=['Random', 'LC', 'Margin', 'Entropy'],
                        help='Type of strategy to use for querying data in active learning process')
    parser.add_argument('--data_aug', action='store_true',
                        help="Data augmentation")
    parser.add_argument('--save_path', type=str, default="./runs", help='The path where the output will be stored,'
                                                                        'model weights as well as the figures of '
                                                                        'experiments')

    return parser.parse_args()


if __name__ == "__main__":
    args = argument_parser()

    val_set = args.validation
    data_augment = args.data_aug

    # Hyper parameters to test
    _batch_sizes = [20, 50, 128]
    _num_epochs = [10, 15, 25, 30]
    _learning_rates = [1e-2, 1e-3, 1e-1, 1, 0.2, 0.3]
    _initial_data_ratios = [0.5, 0.4, 0.2, 0.1]
    _train_set_thresholds = [0.7, 0.8, 0.9, 1.0]
    _query_sizes = [8000, 4000, 2000, 1000, 500]
    _optimizers = ['Adam', 'SGD']
    hyper_params = product(_batch_sizes, _num_epochs, _learning_rates,
                           _initial_data_ratios, _train_set_thresholds, _query_sizes, _optimizers)

    # Loading the data
    train_set, test_set = get_data(data_augment, args.dataset)
    # safely create save path
    check_dir(args.save_path)

    if args.dataset == 'mnistfashion':
        num_channels = 1
        num_classes = 10
    elif args.dataset == 'cifar100':
        num_channels = 3
        num_classes = 100

    if args.model == 'BasicCNN':
        model = SimpleModel(num_channels=num_channels, num_classes=num_classes)
    elif args.model == 'SENet':
        model = SENet(num_channels=num_channels, num_classes=num_classes)
    elif args.model == 'ResNet':
        model = ResNet(num_channels=num_channels, num_classes=num_classes)
    elif args.model == 'DenseNet':
        model = DenseNet(num_channels=num_channels, num_classes=num_classes)

    for batch_size, num_epochs, learning_rate, initial_data_ratio, train_set_threshold, query_size, optimizer in \
            hyper_params:

        hp_str = f'{model.__class__.__name__}' + '_'.join([str(item) for item in [batch_size, num_epochs,
                                                                                  learning_rate, initial_data_ratio,
                                                                                  train_set_threshold,
                                                                                  query_size, optimizer]])
        if val_set is not None:
            dm = DM(train_set, test_set, batch_size=batch_size, validation=val_set,
                    initial_train_dataset_ratio=initial_data_ratio)
        else:
            dm = DM(train_set, test_set, batch_size=batch_size,
                    initial_train_dataset_ratio=initial_data_ratio)

        if args.query_strategy == 'Random':
            query_strategy = RandomQueryStrategy(dm)
        elif args.query_strategy == 'LC':
            query_strategy = LCQueryStrategy(dm)
            pass
        elif args.query_strategy == 'Margin':
            query_strategy = MSQueryStrategy(dm)
            pass
        elif args.query_strategy == 'Entropy':
            query_strategy = EntropyQueryStrategy(dm)

        # adjust num_query with threshold
        num_query = len(train_set) * train_set_threshold * (1 - initial_data_ratio) // query_size

        if optimizer == 'SGD':
            optimizer_factory = optimizer_setup(optim.SGD, lr=learning_rate, momentum=0.9)
        elif optimizer == 'Adam':
            optimizer_factory = optimizer_setup(optim.Adam, lr=learning_rate)

        model_trainer = TrainTestManager(model=model,
                                         querier=query_strategy,
                                         loss_fn=nn.CrossEntropyLoss(),
                                         optimizer_factory=optimizer_factory,
                                         )
        model_trainer.train(num_epochs=num_epochs, num_query=num_query, query_size=query_size)
        # initialize the writer
        writer = SummaryWriter(args.save_path + f"/{hp_str}")
        for i in range(len(model_trainer.metric_values['global_train_loss'])):
            writer.add_scalar('global_test_accuracy', model_trainer.metric_values['global_test_accuracy'][i],
                              model_trainer.metric_values['number_of_data'][i])
            writer.add_scalar('global_test_loss', model_trainer.metric_values['global_test_loss'][i],
                              model_trainer.metric_values['number_of_data'][i])
        writer.close()
        print(f"{hp_str} written ")
