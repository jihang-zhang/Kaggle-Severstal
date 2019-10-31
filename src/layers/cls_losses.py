from __future__ import print_function, division

from include import *
import math
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.autograd import Function
import torch.nn.functional as F
import numpy as np

try:
    from itertools import  ifilterfalse
except ImportError: # py3k
    from itertools import  filterfalse


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, reduction=True):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logit, target):
        target = target.float()
        max_val = (-logit).clamp(min=0)
        loss = logit - logit * target + max_val + \
               ((-max_val).exp() + (-logit - max_val).exp()).log()

        invprobs = F.logsigmoid(-logit * (target * 2.0 - 1.0))
        loss = (invprobs * self.gamma).exp() * loss

        if self.reduction:
            if len(loss.size())==2:
                loss = loss.sum(dim=1)
            return loss.mean()
        else:
            return loss
        

class RingLoss(nn.Module):
    def __init__(self, type='auto', loss_weight=1.0):
        """
        :param type: type of loss ('l1', 'l2', 'auto')
        :param loss_weight: weight of loss, for 'l1' and 'l2', try with 0.01. For 'auto', try with 1.0.
        :return:
        """
        super(RingLoss, self).__init__()
        self.radius = Parameter(torch.Tensor(1))
        self.radius.data.fill_(-1)
        self.loss_weight = loss_weight
        self.type = type

    def forward(self, x):
        x = x.pow(2).sum(dim=1).pow(0.5)
        if self.radius.data[0] < 0: # Initialize the radius with the mean feature norm of first iteration
            self.radius.data.fill_(x.mean().data)
        if self.type == 'l1': # Smooth L1 Loss
            loss1 = F.smooth_l1_loss(x, self.radius.expand_as(x)).mul_(self.loss_weight)
            loss2 = F.smooth_l1_loss(self.radius.expand_as(x), x).mul_(self.loss_weight)
            ringloss = loss1 + loss2
        elif self.type == 'auto': # Divide the L2 Loss by the feature's own norm
            diff = x.sub(self.radius.expand_as(x)) / (x.mean().detach().clamp(min=0.5))
            diff_sq = torch.pow(torch.abs(diff), 2).mean()
            ringloss = diff_sq.mul_(self.loss_weight)
        else: # L2 Loss, if not specified
            diff = x.sub(self.radius.expand_as(x))
            diff_sq = torch.pow(torch.abs(diff), 2).mean()
            ringloss = diff_sq.mul_(self.loss_weight)
        return ringloss

# --------------------------- Metric Learning Losses ---------------------------
class ArcFaceLoss(nn.modules.Module):
    def __init__(self,s=30.0,m=0.5):
        super(ArcFaceLoss, self).__init__()
        self.classify_loss = nn.BCEWithLogitsLoss()
        self.s = s
        self.easy_margin = False
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, logits, labels):
        cosine = logits
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # one_hot = torch.zeros(cosine.size(), device='cuda')
        # one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (labels * phi) + ((1.0 - labels) * cosine)
        output *= self.s
        loss = self.classify_loss(output, labels)
        return loss


#---------------------UDA-----------------------------
class TSA():
    def __init__(self, num_classes, num_train_optimization_steps, mode='linear'):
        if mode == 'linear':
            self.scheduler = lambda x: (x/num_train_optimization_steps) * (1-1/num_classes)+1/num_classes
        if mode == 'log':
            self.scheduler = lambda x: (1-math.exp(-5*x/num_train_optimization_steps)) * (1-1/num_classes)+1/num_classes
        if mode == 'exp':
            self.scheduler = lambda x: math.exp((x/num_train_optimization_steps-1)*5) * (1-1/num_classes)+1/num_classes
        else:
            ValueError("Invalid mode {} - please choose one from ['linear', 'log', 'exp']".format(mode))

    def step(self, t):
        return self.scheduler(t)


def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))


class OhemBCEWithLogitsLoss(nn.Module): 
    def __init__(self, weight=None): 
        super(OhemBCEWithLogitsLoss, self).__init__() 
        self.criterion = nn.BCEWithLogitsLoss(weight=weight, reduction='none') 
    
    def forward(self, score, target, thresh):
        pred = torch.sigmoid(score).contiguous().view(-1)
        losses = self.criterion(score, target).contiguous().view(-1)
        losses = losses[pred < thresh] 
        return losses.mean()


class OhemFocalLoss(nn.Module): 
    def __init__(self, gamma=2): 
        super(OhemFocalLoss, self).__init__() 
        self.criterion = FocalLoss(gamma=gamma, reduction=False) 
    
    def forward(self, score, target, thresh):
        pred = torch.sigmoid(score).contiguous().view(-1)
        losses = self.criterion(score, target).contiguous().view(-1)
        losses = losses[pred < thresh] 
        return losses.mean()


def floss(logits, labels):
    __small_value = 1e-6
    beta = 0.5
    batch_size = logits.size()[0]
    p = torch.sigmoid(logits)
    l = labels
    num_pos = torch.sum(p, 1) + __small_value
    num_pos_hat = torch.sum(l, 1) + __small_value
    tp = torch.sum(l * p, 1)
    precise = tp / num_pos
    recall = tp / num_pos_hat
    fs = (1 + beta * beta) * precise * recall / (beta * beta * precise + recall + __small_value)
    loss = fs.sum() / batch_size
    return (1 - loss)