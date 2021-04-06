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

from query_strats.RandomQueryStrategy import RandomQueryStrategy
from utils import get_data
from DataManager import DataManager as DM


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
    train_set_threshold = args.train_set_threshold
    query_size = args.query_size
    data_augment = args.data_aug

    train_set, test_set = get_data(data_augment, args.dataset)
    dm = DM(train_set, test_set, batch_size=batch_size)
    query_strat = RandomQueryStrategy(dm)

    for i in range(10):
        idx = query_strat.execute_query(query_size)
        query_strat.update_label(idx)


