# -*- coding:utf-8 -*-

"""
University of Sherbrooke
Date:
Authors: Mamadou Mountagha BAH & Pierre-Marc Jodoin
License:
Other: Suggestions are welcome
"""

import torch.nn as nn
from models.CNNBaseModel import CNNBaseModel


class SqueezeBlock(nn.Module):
    """
       this block is the building block of the Squeeze and Excite Net
       it takes an input with in_channels, squeeze input to attention 1D array
       the size of the number of feature maps,
       multiply attention array to original input (scaling by feature maps)
       and sum it up to the original input.
    """

    def __init__(self, in_channels, squeeze_rate, feature_map_size):
        super(SqueezeBlock, self).__init__()
        self.globalpool = nn.AvgPool2d(kernel_size=feature_map_size)  # 1x1xin_channels
        self.FC1 = nn.Linear(in_channels, in_channels//squeeze_rate)
        self.bn1 = nn.BatchNorm1d(in_channels//squeeze_rate)
        self.ReLU = nn.ReLU()
        self.FC2 = nn.Linear(in_channels//squeeze_rate, in_channels)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        attention = self.globalpool(x)
        attention = attention.view(attention.size(0), -1)
        attention = self.ReLU(self.bn1(self.FC1(attention)))
        attention = self.sig(self.FC2(attention))
        attention = attention.view(-1, attention.size(1), 1, 1)
        output = attention*x + x  # Multiply input by attention vector then add input
        return output


class SENet(CNNBaseModel):
    """
        Class that implements the SE-ResNet 18 layers model.
        (Note: for computational efficiency, only one of each blocks described
               in the referenced article are used)
        Reference:
        [1] Jie Hu, Li Shen, Samuel Albanie, Gang Sun, Enhua Wu
        Squeeze-and-Excitation Networks. arXiv:1709.01507
    """
    def __init__(self, num_classes=10, num_channels=1, init_weights=True):
        """
        Builds simplified version of SE-ResNet-50 model.
        Args:
              num_classes(int): number of classes. default 200(tiny imagenet)

              init_weights(bool): when true uses _initialize_weights function to initialize
              network's weights.
        """
        super(SENet, self).__init__(num_classes, init_weights)

        self.initial_block = nn.Sequential(
            nn.Conv2d(in_channels=num_channels, out_channels=64, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256)
        )

        self.stage1 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 512, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            SqueezeBlock(512, squeeze_rate=16, feature_map_size=28)
        )
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.stage2 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 1024, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm2d(1024),
            SqueezeBlock(1024, squeeze_rate=16, feature_map_size=14)
        )
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.stage3 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 2048, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm2d(2048),
            SqueezeBlock(2048, squeeze_rate=16, feature_map_size=7)
        )

        self.final_block = nn.Sequential(
            nn.AvgPool2d(kernel_size=7),  # Global pooling
            nn.Flatten(),
            nn.Linear(2048, 10)
        )

    def forward(self, x):
        output = self.initial_block(x)
        output = self.stage1(output)
        output = self.maxpool1(output)
        output = self.stage2(output)
        output = self.maxpool2(output)
        output = self.stage3(output)
        output = self.final_block(output)
        return output
