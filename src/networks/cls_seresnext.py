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
import pretrainedmodels

class SEResNeXtMultiCLS(nn.Module):
    def __init__(self, feature_net='se_resnext50_32x4d', imagenet_pretrained=True, num_classes=4):
        super().__init__()

        if imagenet_pretrained == True:
            feature_net = pretrainedmodels.__dict__[feature_net](num_classes=1000, pretrained='imagenet')
        else:
            feature_net = pretrainedmodels.__dict__[feature_net](num_classes=num_classes, pretrained=None)

        self.blocks = nn.Sequential(
            feature_net.layer0,
            feature_net.layer1,
            feature_net.layer2,
            feature_net.layer3,
            feature_net.layer4)

        feature_net = None

        self.dropout = nn.Dropout(0.5)
        self.last_linear = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.blocks(x)

        x = F.adaptive_avg_pool2d(x, 1)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x