from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import logging
import functools

import numpy as np

import torch
import torch.nn as nn
import torch._utils
import torch.nn.functional as F

sys.path.insert(0, '../layers')
from backbone.resnet import *


class ResNetMultiCLS(nn.Module):
    def __init__(self, feature_net='resnet34', imagenet_pretrained=True, num_classes=4, drop=0.5):
        super().__init__()

        if feature_net == 'resnet18':
            resnet = resnet18(pretrained=imagenet_pretrained)
            self.EX = 1
        elif feature_net == 'resnet34':
            resnet = resnet34(pretrained=imagenet_pretrained)
            self.EX = 1
        elif feature_net == 'resnet50':
            resnet = resnet50(pretrained=imagenet_pretrained)
            self.EX = 4
        elif feature_net == 'resnet101':
            resnet = resnet101(pretrained=imagenet_pretrained)
            self.EX = 4
        elif feature_net == 'resnet152':
            resnet = resnet152(pretrained=imagenet_pretrained)
            self.EX = 4

        self.blocks = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4)

        resnet = None

        self.dropout = nn.Dropout(drop)
        self.feature = nn.Conv2d(512 * self.EX, 32, kernel_size=1)
        self.logit = nn.Conv2d(32, num_classes, kernel_size=1)

    def freeze_bn_dropout(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
            if isinstance(m, nn.Dropout):
                m.eval()

    def forward(self, x):
        x = self.blocks(x)

        x = F.adaptive_avg_pool2d(x, 1)
        x = self.dropout(x)
        x = self.feature(x)
        x = self.logit(x).squeeze()
        return x


class ResNetMulti(nn.Module):
    def __init__(self, feature_net='resnet34', imagenet_pretrained=True, num_classes=4):
        super().__init__()

        if feature_net == 'resnet18':
            self.resnet = resnet18(pretrained=imagenet_pretrained, num_classes=num_classes)
            self.EX = 1
        elif feature_net == 'resnet34':
            self.resnet = resnet34(pretrained=imagenet_pretrained, num_classes=num_classes)
            self.EX = 1
        elif feature_net == 'resnet50':
            self.resnet = resnet50(pretrained=imagenet_pretrained, num_classes=num_classes)
            self.EX = 4
        elif feature_net == 'resnet101':
            self.resnet = resnet101(pretrained=imagenet_pretrained, num_classes=num_classes)
            self.EX = 4
        elif feature_net == 'resnet152':
            self.resnet = resnet152(pretrained=imagenet_pretrained, num_classes=num_classes)
            self.EX = 4

    def forward(self, x):
        x = self.resnet(x)

        return x