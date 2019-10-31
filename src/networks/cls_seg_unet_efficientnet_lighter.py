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
import segmentation_models_pytorch as smp


class EfficientUnet(nn.Module):

    def __init__(self, encoder_type='efficientnet-b3',
                 dropout_p=0.5,
                 attention_type=None,
                 reduction=16,
                 reslink=False,
                 out_channel=1,
                 imagenet_pretrained=True
                 ):
        super().__init__()

        self.attention = attention_type is not None
        self.attention_type = attention_type
        self.out_channel = out_channel
        decoder_kernels = [1, 1, 1, 1]

        if imagenet_pretrained == True:
            self.encoder = smp.encoders.get_encoder(encoder_type, encoder_weights='imagenet') 
        else:
            self.encoder = smp.encoders.get_encoder(encoder_type, encoder_weights=None) 

        att_type=self.attention_type
        self.decoder4 = Decoder(self.encoder.out_shapes[1] + 32, self.encoder.out_shapes[1], 32,
                                attention_type=att_type,
                                attention_kernel_size=decoder_kernels[1],
                                reduction=reduction,
                                reslink=reslink)
        self.decoder3 = Decoder(self.encoder.out_shapes[2] + 32, self.encoder.out_shapes[2], 32,
                                attention_type=att_type,
                                attention_kernel_size=decoder_kernels[2],
                                reduction=reduction,
                                reslink=reslink)
        self.decoder2 = Decoder(self.encoder.out_shapes[3] + 32, self.encoder.out_shapes[3], 32,
                                up_sample=False,
                                attention_type=att_type,
                                attention_kernel_size=decoder_kernels[3],
                                reduction=reduction,
                                reslink=reslink)

        self.logit = nn.Sequential(
            ConvBnRelu2d(128, 32, kernel_size=3, padding=1),
            nn.Conv2d(32, out_channel, kernel_size=1, padding=0),
        )

        center_channels = self.encoder.out_shapes[0]
        decoder5_channels = self.encoder.out_shapes[0] + 128

        self.center = nn.Sequential(
            ConvBn2d(center_channels, center_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            ConvBn2d(center_channels, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.decoder5 = Decoder(decoder5_channels, self.encoder.out_shapes[0], 32,
                                attention_type=att_type,
                                attention_kernel_size=decoder_kernels[0],
                                reduction=reduction,
                                reslink=reslink)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=dropout_p)
        self.last_linear = nn.Linear(center_channels, out_channel)

        feature_net = None

    def forward(self, x, *args):

        e5, e4, e3, e2, _ = self.encoder(x)

        f = self.center(e5)

        d5 = self.decoder5(torch.cat([f, e5], 1))
        d4 = self.decoder4(torch.cat([d5, e4], 1))
        d3 = self.decoder3(torch.cat([d4, e3], 1))
        d2 = self.decoder2(torch.cat([d3, e2], 1))
        f = torch.cat((
                 d2,
                 d3,
                 F.interpolate(d4, scale_factor=2, mode='bilinear', align_corners=False),
                 F.interpolate(d5, scale_factor=4, mode='bilinear', align_corners=False),
        ), 1)
        logit_pixel = self.logit(f)

        out = self.avg_pool(e5)
        out = self.dropout(out)
        out = out.view(out.size(0), -1)
        logit_image = self.last_linear(out)
        return logit_pixel, logit_image