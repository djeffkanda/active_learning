# -*- coding:utf-8 -*-

"""
University of Sherbrooke
NN Class Project
Authors: D'Jeff Kanda, Gabriel McCarthy, Mohamed Ragued
"""

import torch.nn as nn
from models.CNNBaseModel import CNNBaseModel


class SimpleModel(CNNBaseModel):
    """
    Class that implements a very basic model conv-relu-bn x3 +2 FC.
    """

    def __init__(self, num_classes=10, num_channels=1, init_weights=True):
        """
        Builds SimpleModel.
        Args:
            num_classes(int): number of classes. default 10(cifar 10)
            init_weights(bool): when true uses _initialize_weights function to initialize
            network's weights.
        """
        super(SimpleModel, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=num_channels, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(256 * 4 * 4, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        """
        Forward pass of the model
        Args:
            x: Tensor
        """
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # reshape feature maps
        x = self.fc_layers(x)
        return x
