#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
University of Sherbrooke
Date:
Authors: Mamadou Mountagha BAH & Pierre-Marc Jodoin
License: Opensource, free to use
Other: Suggestions are welcome
"""

import argparse
import torch.optim as optim
import torch.nn as nn

from utils import get_data
from query_strats.DataManager import DataManager as DM
from TrainTestManager import TrainTestManager, optimizer_setup

from models.SimpleModel import SimpleModel
from query_strats.RandomQueryStrategy import RandomQueryStrategy


def argument_parser():
    """
        A parser to allow user to easily experiment different models along with datasets and differents parameters
    """
    parser = argparse.ArgumentParser(usage='\n python3 train.py [model] [dataset] [hyper_parameters]'
                                           '\n python3 train.py --model SENet [hyper_parameters]'
                                           '\n python3 train.py --model SENet --predict',
                                     description="This program allows to train different models of classification on"
                                                 " different datasets. Active learning is used to prioritise sample "
                                                 "selection for in the training process"
                                     )
    parser.add_argument('--model', type=str, default="BasicCNN",
                        choices=["BasicCNN", "SENet"])
    parser.add_argument('--dataset', type=str, default="mnistfashion", choices=["cifar100", "mnistfashion"])
    parser.add_argument('--batch_size', type=int, default=20,
                        help='The size of the training batch')
    parser.add_argument('--optimizer', type=str, default="Adam", choices=["Adam", "SGD"],
                        help="The optimizer to use for training the model")
    parser.add_argument('--num-epochs', type=int, default=10,
                        help='The number of epochs')
    parser.add_argument('--validation', type=float, default=0.1,
                        help='Percentage of training data to use for validation')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--initial_data_ratio', type=float, default=0.2,
                        help='Percentage of training data randomly selected on first iteration of active'
                             'learning process')
    parser.add_argument('--query_strategy', type=str, default='Random',
                        choices=['Random', 'Uncertainty', 'Margin', 'Entropy'],
                        help='Type of strategy to use for querying data in active learning process')
    parser.add_argument('--query_size', type=int, default=200,
                        help='Size of sample to label per query')
    parser.add_argument('--train_set_threshold', type=float, default=1,
                        help='Percentage of training data as threshold to stop active learning process')
    parser.add_argument('--data_aug', action='store_true',
                        help="Data augmentation")
    parser.add_argument('--save_path', type=str, default="./", help='The path where the output will be stored,'
                                                                    'model weights as well as the figures of '
                                                                    'experiments')
    return parser.parse_args()


if __name__ == "__main__":
    args = argument_parser()

    batch_size = args.batch_size
    num_epochs = args.num_epochs
    val_set = args.validation
    learning_rate = args.lr
    initial_data_ratio = args.initial_data_ratio
    train_set_threshold = args.train_set_threshold
    query_size = args.query_size
    data_augment = args.data_aug

    # Loading the data
    train_set, test_set = get_data(data_augment, args.dataset)
    if val_set is not None:
        dm = DM(train_set, test_set, batch_size=batch_size, validation=val_set,
                initial_train_dataset_ratio=initial_data_ratio)
    else:
        dm = DM(train_set, test_set, batch_size=batch_size,
                initial_train_dataset_ratio=initial_data_ratio)

    if args.optimizer == 'SGD':
        optimizer_factory = optimizer_setup(optim.SGD, lr=learning_rate, momentum=0.9)
    elif args.optimizer == 'Adam':
        optimizer_factory = optimizer_setup(optim.Adam, lr=learning_rate)

    if args.dataset == 'mnistfashion':
        num_channels = 1
        num_classes = 10
    elif args.dataset == 'cifar100':
        num_channels = 3
        num_classes = 100

    if args.model == 'BasicCNN':
        model = SimpleModel(num_channels=num_channels, num_classes=num_classes)
    elif args.model == 'SENet':
        # model = SENet(num_channels=num_channels, num_classes=num_classes))
        pass
    elif args.model == 'ResNet':
        # model = ResNet(num_channels=num_channels, num_classes=num_classes))
        pass

    if args.query_strategy == 'Random':
        query_strategy = RandomQueryStrategy(dm)
    elif args.query_strategy == 'Uncertainty':
        #  query_strategy = UncertaintyQueryStrategy(dm)
        pass
    elif args.query_strategy == 'Margin':
        #  query_strategy = MarginQueryStrategy(dm)
        pass
    elif args.query_strategy == 'Entropy':
        #  query_strategy = EntropyQueryStrategy(dm)
        pass

    model_trainer = TrainTestManager(model=model,
                                     querier=query_strategy,
                                     trainset=train_set,
                                     testset=test_set,
                                     loss_fn=nn.CrossEntropyLoss(),
                                     batch_size=batch_size,
                                     optimizer_factory=optimizer_factory,
                                     validation=val_set)

    model_trainer.train(num_epochs=num_epochs, num_query=5, query_size=query_size)
    # TODO adjust num_query with threshold


