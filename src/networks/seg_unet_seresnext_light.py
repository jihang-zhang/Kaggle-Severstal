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
from decoder import *
import pretrainedmodels


class SEResnextUnet(nn.Module):

    def __init__(self,feature_net='se_resnext50_32x4d',
                 attention_type=None,
                 reduction=16,
                 reslink=False,
                 out_channel=1,
                 imagenet_pretrained=True
                 ):
        super().__init__()
        self.EX = 4
        self.attention = attention_type is not None
        self.attention_type = attention_type
        self.out_channel = out_channel
        decoder_kernels = [1, 1, 1, 1]

        if imagenet_pretrained == True:
            feature_net = pretrainedmodels.__dict__[feature_net](num_classes=1000, pretrained='imagenet')
        else:
            feature_net = pretrainedmodels.__dict__[feature_net](num_classes=1000, pretrained=None)

        self.conv1 = feature_net.layer0
        self.encoder2 = feature_net.layer1
        self.encoder3 = feature_net.layer2
        self.encoder4 = feature_net.layer3
        self.encoder5 = feature_net.layer4
        feature_net = None

        att_type=self.attention_type
        self.decoder4 = Decoder(256*self.EX + 64, 256, 64,
                                attention_type=att_type,
                                attention_kernel_size=decoder_kernels[1],
                                reduction=reduction,
                                reslink=reslink)
        self.decoder3 = Decoder(128*self.EX + 64, 128, 64,
                                attention_type=att_type,
                                attention_kernel_size=decoder_kernels[2],
                                reduction=reduction,
                                reslink=reslink)
        self.decoder2 = Decoder(64*self.EX + 64, 64, 64,
                                attention_type=att_type,
                                attention_kernel_size=decoder_kernels[3],
                                reduction=reduction,
                                reslink=reslink)

        self.logit = nn.Sequential(
            ConvBnRelu2d(256, 64, kernel_size=3, padding=1),
            nn.Conv2d(64, out_channel, kernel_size=1, padding=0),
        )

        center_channels = 512 * self.EX
        decoder5_channels = 512 * self.EX + 256

        self.center = nn.Sequential(
            ConvBn2d(center_channels, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            ConvBn2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.decoder5 = Decoder(decoder5_channels, 512, 64,
                                attention_type=att_type,
                                attention_kernel_size=decoder_kernels[0],
                                reduction=reduction,
                                reslink=reslink)

    def forward(self, x, *args):
        x = self.conv1(x)

        e2 = self.encoder2(x)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        e5 = self.encoder5(e4)

        f = self.center(e5)

        d5 = self.decoder5(torch.cat([f, e5], 1))
        d4 = self.decoder4(torch.cat([d5, e4], 1))
        d3 = self.decoder3(torch.cat([d4, e3], 1))
        d2 = self.decoder2(torch.cat([d3, e2], 1))
        f = torch.cat((
                 d2,
                 F.interpolate(d3, scale_factor=2, mode='bilinear', align_corners=False),
                 F.interpolate(d4, scale_factor=4, mode='bilinear', align_corners=False),
                 F.interpolate(d5, scale_factor=8, mode='bilinear', align_corners=False),
        ), 1)
        logit = self.logit(f)
        return logit