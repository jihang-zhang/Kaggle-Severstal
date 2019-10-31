import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

class SeparableConv2d(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, dilation=1, relu_first=True):
        super().__init__()
        depthwise = nn.Conv2d(inplanes, inplanes, kernel_size,
                              stride=stride, padding=dilation,
                              dilation=dilation, groups=inplanes, bias=False)
        bn_depth = nn.BatchNorm2d(inplanes)
        pointwise = nn.Conv2d(inplanes, planes, 1, bias=False)
        bn_point = nn.BatchNorm2d(planes)

        if relu_first:
            self.block = nn.Sequential(OrderedDict([('relu', nn.ReLU()),
                                                    ('depthwise', depthwise),
                                                    ('bn_depth', bn_depth),
                                                    ('pointwise', pointwise),
                                                    ('bn_point', bn_point)
                                                    ]))
        else:
            self.block = nn.Sequential(OrderedDict([('depthwise', depthwise),
                                                    ('bn_depth', bn_depth),
                                                    ('relu1', nn.ReLU()),
                                                    ('pointwise', pointwise),
                                                    ('bn_point', bn_point),
                                                    ('relu2', nn.ReLU())
                                                    ]))

    def forward(self, x):
        return self.block(x)


class ASPP(nn.Module):
    def __init__(self, in_channels=384, out_channels=256, output_stride=8):
        super().__init__()
        if output_stride == 16:
            dilations = [6, 12, 18]
        elif output_stride == 8:
            dilations = [12, 24, 36]
        else:
            raise NotImplementedError
        # dilations = [6, 12, 18]

        self.aspp0 = nn.Sequential(OrderedDict([('conv', nn.Conv2d(in_channels, out_channels, 1, bias=False)),
                                                ('bn', nn.BatchNorm2d(out_channels)),
                                                ('relu', nn.ReLU(inplace=True))]))
        self.aspp1 = SeparableConv2d(in_channels, out_channels, dilation=dilations[0], relu_first=False)
        self.aspp2 = SeparableConv2d(in_channels, out_channels, dilation=dilations[1], relu_first=False)
        self.aspp3 = SeparableConv2d(in_channels, out_channels, dilation=dilations[2], relu_first=False)

        self.image_pooling = nn.Sequential(OrderedDict([('gap', nn.AdaptiveAvgPool2d((1, 1))),
                                                        ('conv', nn.Conv2d(in_channels, out_channels, 1, bias=False)),
                                                        ('bn', nn.BatchNorm2d(out_channels)),
                                                        ('relu', nn.ReLU(inplace=True))]))

        self.dropout = nn.Dropout2d(p=0.1)
        self.conv = nn.Conv2d(out_channels*5, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        pool = self.image_pooling(x)
        pool = F.interpolate(pool, size=x.shape[2:], mode='bilinear', align_corners=True)

        x0 = self.aspp0(x)
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x = torch.cat((pool, x0, x1, x2, x3), dim=1)
        x = self.dropout(x)

        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        
        return x


class JPU(nn.Module):
    def __init__(self, in_channels, width=512):
        super(JPU, self).__init__()

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels[-1], width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels[-2], width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels[-3], width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True))

        self.dilation1 = SeparableConv2d(3*width, width, kernel_size=3, dilation=1, relu_first=False)
        self.dilation2 = SeparableConv2d(3*width, width, kernel_size=3, dilation=2, relu_first=False)
        self.dilation3 = SeparableConv2d(3*width, width, kernel_size=3, dilation=4, relu_first=False)
        self.dilation4 = SeparableConv2d(3*width, width, kernel_size=3, dilation=8, relu_first=False)

    def forward(self, *inputs):
        feats = [self.conv5(inputs[-1]), self.conv4(inputs[-2]), self.conv3(inputs[-3])]
        _, _, h, w = feats[-1].size()
        feats[-2] = F.interpolate(feats[-2], (h, w), mode='bilinear', align_corners=True)
        feats[-3] = F.interpolate(feats[-3], (h, w), mode='bilinear', align_corners=True)
        feat = torch.cat(feats, dim=1)
        feat = torch.cat([self.dilation1(feat), self.dilation2(feat), self.dilation3(feat), self.dilation4(feat)], dim=1)

        return inputs[0], inputs[1], inputs[2], feat



def try_index(scalar_or_list, i):
    try:
        return scalar_or_list[i]
    except TypeError:
        return scalar_or_list


class MiniDL(nn.Module):
    def __init__(self, in_channels, out_channels, dilation, pooling_size):
        super(MiniDL, self).__init__()
        self.pooling_size = pooling_size

        self.conv1_3x3 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.conv1_dil = nn.Conv2d(in_channels, out_channels, 3, dilation=dilation, padding=dilation, bias=False)
        self.conv1_glb = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn1 = nn.Sequential(
            nn.BatchNorm2d(out_channels * 3),
            nn.ReLU(inplace=True))

        self.conv2 = nn.Conv2d(out_channels * 3, out_channels, 1, bias=False)
        self.bn2 = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

    def _global_pooling(self, x):
        pooling_size = (min(try_index(self.pooling_size, 0), x.shape[2]),
                        min(try_index(self.pooling_size, 1), x.shape[3]))
        padding = (
            (pooling_size[1] - 1) // 2,
            (pooling_size[1] - 1) // 2 if pooling_size[1] % 2 == 1 else (pooling_size[1] - 1) // 2 + 1,
            (pooling_size[0] - 1) // 2,
            (pooling_size[0] - 1) // 2 if pooling_size[0] % 2 == 1 else (pooling_size[0] - 1) // 2 + 1
        )

        pool = F.avg_pool2d(x, pooling_size, stride=1)
        pool = F.pad(pool, pad=padding, mode="replicate")
        return pool

    def forward(self, x):
        x = torch.cat([
            self.conv1_3x3(x),
            self.conv1_dil(x),
            self.conv1_glb(self._global_pooling(x)),
        ], dim=1)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        return x
