# -*- coding:utf-8 -*-
"""
University of Sherbrooke
NN Class Project
Authors: D'Jeff Kanda, Gabriel McCarthy, Mohamed Ragued
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.CNNBaseModel import CNNBaseModel


class DenseLayer(nn.Module):
    """
    this block is the building block of the dense network. it takes an
    input with in_channels and applies 4 blocks of convolutional layers.
    Each convolutional block takes the concatenation of all previous feature maps
    inside the dense block as input.
    """

    def __init__(self, in_channels, bottleneck_channels, concat_block_size=32):
        super(DenseLayer, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, bottleneck_channels, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(bottleneck_channels)
        self.conv2 = nn.Conv2d(bottleneck_channels, concat_block_size, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(concat_block_size)

    def forward(self, x):

        output = F.relu(self.bn1(self.conv1(x)))
        output = F.relu(self.bn2(self.conv2(output)))
        return torch.cat([x, output], 1)


class DenseBlock(nn.Module):
    """
    this block is the building block of the dense network. it takes an
    input with in_channels and applies 4 blocks of convolutional layers.
    Each convolutional block takes the concatenation of all previous feature maps
    inside the dense block as input.
    """

    def __init__(self, in_channels, bottleneck_channels, num_dense_layer, concat_block_size=32):
        super(DenseBlock, self).__init__()

        layers = []
        for i in range(num_dense_layer):
            layers.append(
                DenseLayer(in_channels + i * concat_block_size, bottleneck_channels, concat_block_size)
            )

        self.block_layer = nn.Sequential(*layers)

    def forward(self, x):
        return self.block_layer(x)


class DenseNet(CNNBaseModel):
    """
        Class that implements simplified DenseNet.
        Reference:
        [1] Gao Huang, Zhuang Liu, Laurens van der Maaten, Kilian Q. Weinberger
        Densely Connected Convolutional Networks. arXiv:1608.06993
        """
    def __init__(self, num_classes=10, num_channels=1, init_weights=True):
        """
        Builds simplified model inspired by DenseNet model.
            Args:
            num_classes(int): number of classes. default 10(fashionmnist)
            num_channels(int): number of channel for input image. default 1(fashionmnist)
            init_weights(bool): when true uses _initialize_weights function to initialize
            network's weights.
        """
        super(DenseNet, self).__init__(num_classes, init_weights)
        self.initial_layers = nn.Sequential(
            nn.Conv2d(in_channels=num_channels, out_channels=64, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(64)
        )

        self.dense_layers = nn.Sequential(
            DenseBlock(in_channels=64, bottleneck_channels=128, num_dense_layer=6, concat_block_size=32),
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            DenseBlock(in_channels=128, bottleneck_channels=128, num_dense_layer=12, concat_block_size=32),
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            DenseBlock(in_channels=256, bottleneck_channels=128, num_dense_layer=24, concat_block_size=32),
            nn.AvgPool2d(kernel_size=7)  # Global Pooling
        )

        self.final_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        output = self.initial_layers(x)
        output = self.dense_layers(output)
        output = self.final_layers(output)
        return output
