from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import logging
import functools
from collections import OrderedDict

import numpy as np

import torch
import torch.nn as nn
import torch._utils
import torch.nn.functional as F

sys.path.insert(0, '../layers')
from backbone.resnet import *
from seg_layers import *


def upsample_add(x, lateral):
    return F.relu(F.interpolate(x, size=lateral.shape[2:], mode='nearest') + lateral, inplace=True)


class FPN(nn.Module):

    def __init__(self,feature_net='resnet34',
                 out_channel=1,
                 imagenet_pretrained=True
                 ):
        super().__init__()

        self.out_channel = out_channel

        if feature_net == 'resnet18':
            self.resnet = resnet18(pretrained=imagenet_pretrained)
            self.EX = 1
        elif feature_net == 'resnet34':
            self.resnet = resnet34(pretrained=imagenet_pretrained)
            self.EX = 1
        elif feature_net == 'resnet50':
            self.resnet = resnet50(pretrained=imagenet_pretrained)
            self.EX = 4
        elif feature_net == 'resnet101':
            self.resnet = resnet101(pretrained=imagenet_pretrained)
            self.EX = 4
        elif feature_net == 'resnet152':
            self.resnet = resnet152(pretrained=imagenet_pretrained)
            self.EX = 4

        self.conv1 = nn.Sequential(
            self.resnet.conv1,
            self.resnet.bn1,
            self.resnet.relu,
            self.resnet.maxpool
        )
        self.encoder2 = self.resnet.layer1
        self.encoder3 = self.resnet.layer2
        self.encoder4 = self.resnet.layer3
        self.encoder5 = self.resnet.layer4

        self.lateral5 = self._make_lateral(512*self.EX, 256)
        self.lateral4 = self._make_lateral(256*self.EX, 256)
        self.lateral3 = self._make_lateral(128*self.EX, 256)
        self.lateral2 = self._make_lateral( 64*self.EX, 256)

        self.output5 = self._make_output(256)
        self.output4 = self._make_output(256)
        self.output3 = self._make_output(256)
        self.output2 = self._make_output(256)

    @staticmethod
    def _make_lateral(input_channels, hidden_channels):
        return nn.Sequential(OrderedDict([
            ("conv", nn.Conv2d(input_channels, hidden_channels, 1, bias=False)),
            ("bn", nn.BatchNorm2d(hidden_channels))
        ]))


    @staticmethod
    def _make_output(channels):
        return nn.Sequential(OrderedDict([
            ("conv", nn.Conv2d(channels, channels, 3, padding=1, bias=False)),
            ("bn", nn.BatchNorm2d(channels)),
            ("relu", nn.ReLU(inplace=True))
        ]))

    def forward(self, x):

        x = self.conv1(x)

        c2 = self.encoder2(x)
        c3 = self.encoder3(c2)
        c4 = self.encoder4(c3)
        c5 = self.encoder5(c4)

        p5_1x1 = self.lateral5(c5)
        p4_1x1 = upsample_add(p5_1x1, self.lateral4(c4))
        p3_1x1 = upsample_add(p4_1x1, self.lateral3(c3))
        p2_1x1 = upsample_add(p3_1x1, self.lateral2(c2))

        p5 = self.output5(p5_1x1)
        p4 = self.output4(p4_1x1)
        p3 = self.output3(p3_1x1)
        p2 = self.output2(p2_1x1)

        return p5, p4, p3, p2
