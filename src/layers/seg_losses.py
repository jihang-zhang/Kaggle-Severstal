"""
Lovasz-Softmax and Jaccard hinge loss in PyTorch
Maxim Berman 2018 ESAT-PSI KU Leuven (MIT License)
"""

from __future__ import print_function, division

import sys
from typing import List
import math
import torch
import torch.nn as nn
from torch import Tensor, einsum
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from functools import partial
try:
    from itertools import  ifilterfalse
except ImportError: # py3k
    from itertools import  filterfalse as ifilterfalse

from surface_loss_utils import simplex, one_hot
from cls_losses import *

sys.path.insert(0, '../src/utils')
from utils.pytorch_utils import make_one_hot_binary, make_one_hot_multiclass


def fuseloss(score, target):
    lovasz = LovaszSoftmax()
    ohemce = OhemCrossEntropy()
    return ohemce(score, target) + lovasz(score, target)


class CrossEntropy(nn.Module):
    def __init__(self, ignore_label=-1, weight=None):
        super(CrossEntropy, self).__init__()
        self.ignore_label = ignore_label
        self.criterion = nn.CrossEntropyLoss(weight=weight, 
                                             ignore_index=ignore_label)

    def forward(self, score, target):
        ph, pw = score.size(2), score.size(3)
        h, w = target.size(1), target.size(2)
        if ph != h or pw != w:
            score = F.interpolate(input=score, size=(h, w), mode='bilinear', align_corners=False)

        loss = self.criterion(score, target)

        return loss

class OhemCrossEntropy(nn.Module): 
    def __init__(self, ignore_label=-1, thres=0.7, 
        min_kept=100000, weight=None): 
        super(OhemCrossEntropy, self).__init__() 
        self.thresh = thres
        self.min_kept = max(1, min_kept)
        self.ignore_label = ignore_label 
        self.criterion = nn.CrossEntropyLoss(weight=weight, 
                                             ignore_index=ignore_label, 
                                             reduction='none') 
    
    def forward(self, score, target, **kwargs):
        ph, pw = score.size(2), score.size(3)
        h, w = target.size(1), target.size(2)
        if ph != h or pw != w:
            score = F.interpolate(input=score, size=(h, w), mode='bilinear', align_corners=False)
        pred = F.softmax(score, dim=1)
        pixel_losses = self.criterion(score, target).contiguous().view(-1)
        mask = target.contiguous().view(-1) != self.ignore_label         
        
        tmp_target = target.clone() 
        tmp_target[tmp_target == self.ignore_label] = 0 
        pred = pred.gather(1, tmp_target.unsqueeze(1)) 
        pred, ind = pred.contiguous().view(-1,)[mask].contiguous().sort()
        min_value = pred[min(self.min_kept, pred.numel() - 1)] 
        threshold = max(min_value, self.thresh) 
        
        pixel_losses = pixel_losses[mask][ind]
        pixel_losses = pixel_losses[pred < threshold] 
        return pixel_losses.mean()


class LovaszSoftmax(nn.Module):
    def __init__(self, ignore_label=-1): 
        super(LovaszSoftmax, self).__init__()
        self.ignore_label = ignore_label
        self.criterion = partial(lovasz_softmax, ignore=ignore_label)

    def forward(self, score, target):
        ph, pw = score.size(2), score.size(3)
        h, w = target.size(1), target.size(2)
        if ph != h or pw != w:
            score = F.interpolate(input=score, size=(h, w), mode='bilinear', align_corners=False)

        pred = F.softmax(score, dim=1)
        loss = self.criterion(pred, target)

        return loss


class MultiLabelSymmetricLovaszHinge(nn.Module):
    def __init__(self, label_weight=None): 
        super(MultiLabelSymmetricLovaszHinge, self).__init__()
        self.label_weight = label_weight

    def forward(self, score, target):
        target = make_one_hot_binary(target, 5)

        assert score.size(1) == target.size(1)

        ph, pw = score.size(2), score.size(3)
        _, c, h, w = target.size()
        if ph != h or pw != w:
            score = F.interpolate(input=score, size=(h, w), mode='bilinear', align_corners=False)

        loss = 0
        for i in range(c):
            cls_loss = (lovasz_hinge(score[:, i, :, :], target[:, i, :, :]) + lovasz_hinge(-score[:, i, :, :], 1 - target[:, i, :, :])) / 2
            if self.label_weight:
                cls_loss *= self.label_weight[i]
            loss += cls_loss

        return loss


class MultiLabelSymmetricLovaszPerImg(nn.Module):
    def __init__(self, label_weight=None): 
        super(MultiLabelSymmetricLovaszPerImg, self).__init__()
        self.label_weight = label_weight

    def forward(self, score, target):
        target = make_one_hot_binary(target, 5)

        assert score.size(1) == target.size(1)

        ph, pw = score.size(2), score.size(3)
        _, c, h, w = target.size()
        if ph != h or pw != w:
            score = F.interpolate(input=score, size=(h, w), mode='bilinear', align_corners=False)

        loss = []
        for i in range(c):
            cls_loss = (lovasz_per_img(score[:, i, :, :], target[:, i, :, :]) + \
                lovasz_per_img(-score[:, i, :, :], 1 - target[:, i, :, :])) / 2
            if self.label_weight:
                cls_loss *= self.label_weight[i]
            loss.append(cls_loss)

        return loss


class SeverstalClsSegLoss(nn.Module):
    def __init__(self, label_weight=None):
        super(SeverstalClsSegLoss, self).__init__()
        self.criterion_img = nn.BCEWithLogitsLoss(pos_weight=label_weight)
        self.criterion_pxl = MultiLabelSymmetricLovaszPerImg(label_weight=label_weight)

    def forward(self, logit_pxl, logit_img, target_pxl, target_img):
        loss_img = self.criterion_img(logit_img, target_img)

        loss_pxl_0, loss_pxl_1, loss_pxl_2, loss_pxl_3 = self.criterion_pxl(logit_pxl, target_pxl)

        loss_pxl_0 = loss_pxl_0 * target_img[:, 0:1]
        loss_pxl_1 = loss_pxl_1 * target_img[:, 1:2]
        loss_pxl_2 = loss_pxl_2 * target_img[:, 2:3]
        loss_pxl_3 = loss_pxl_3 * target_img[:, 3:4]

        npos_0 = target_img[:, 0].sum()
        npos_1 = target_img[:, 1].sum()
        npos_2 = target_img[:, 2].sum()
        npos_3 = target_img[:, 3].sum()

        loss_pxl_0 = loss_pxl_0.sum() / npos_0 if npos_0.item() != 0 else loss_pxl_0.sum()
        loss_pxl_1 = loss_pxl_1.sum() / npos_1 if npos_1.item() != 0 else loss_pxl_1.sum()
        loss_pxl_2 = loss_pxl_2.sum() / npos_2 if npos_2.item() != 0 else loss_pxl_2.sum()
        loss_pxl_3 = loss_pxl_3.sum() / npos_3 if npos_3.item() != 0 else loss_pxl_3.sum()

        return loss_pxl_0, loss_pxl_1, loss_pxl_2, loss_pxl_3, loss_img


class SeverstalClsSegLossV2(nn.Module):
    def __init__(self, label_weight=None):
        super(SeverstalClsSegLossV2, self).__init__()
        self.criterion_img = nn.BCEWithLogitsLoss(pos_weight=label_weight)
        self.criterion_pxl = MultiLabelSymmetricLovaszPerImg(label_weight=label_weight)

    def forward(self, logit_pxl, logit_img, target_pxl, target_img):
        loss_img = self.criterion_img(logit_img, target_img)

        loss_pxl_0, loss_pxl_1, loss_pxl_2, loss_pxl_3 = self.criterion_pxl(logit_pxl, target_pxl)

        loss_pxl_0 = loss_pxl_0 * target_img[:, 0:1]
        loss_pxl_1 = loss_pxl_1 * target_img[:, 1:2]
        # loss_pxl_2 = loss_pxl_2 * target_img[:, 2:3]
        loss_pxl_3 = loss_pxl_3 * target_img[:, 3:4]

        npos_0 = target_img[:, 0].sum()
        npos_1 = target_img[:, 1].sum()
        # npos_2 = target_img[:, 2].sum()
        npos_3 = target_img[:, 3].sum()

        loss_pxl_0 = loss_pxl_0.sum() / npos_0 if npos_0.item() != 0 else loss_pxl_0.sum()
        loss_pxl_1 = loss_pxl_1.sum() / npos_1 if npos_1.item() != 0 else loss_pxl_1.sum()
        # loss_pxl_2 = loss_pxl_2.sum() / npos_2 if npos_2.item() != 0 else loss_pxl_2.sum()
        loss_pxl_2 = loss_pxl_2.mean()
        loss_pxl_3 = loss_pxl_3.sum() / npos_3 if npos_3.item() != 0 else loss_pxl_3.sum()

        return loss_pxl_0, loss_pxl_1, loss_pxl_2, loss_pxl_3, loss_img


class SeverstalClsSegLossV3(nn.Module):
    def __init__(self, gamma=2, label_weight=None):
        super(SeverstalClsSegLossV3, self).__init__()
        self.criterion_img = OhemFocalLoss(gamma=gamma)
        self.criterion_pxl = MultiLabelSymmetricLovaszPerImg(label_weight=label_weight)

    def forward(self, logit_pxl, logit_img, target_pxl, target_img, eta):
        loss_img = self.criterion_img(logit_img, target_img, eta)

        loss_pxl_0, loss_pxl_1, loss_pxl_2, loss_pxl_3 = self.criterion_pxl(logit_pxl, target_pxl)

        loss_pxl_0 = loss_pxl_0 * target_img[:, 0:1]
        loss_pxl_1 = loss_pxl_1 * target_img[:, 1:2]
        # loss_pxl_2 = loss_pxl_2 * target_img[:, 2:3]
        loss_pxl_3 = loss_pxl_3 * target_img[:, 3:4]

        npos_0 = target_img[:, 0].sum()
        npos_1 = target_img[:, 1].sum()
        # npos_2 = target_img[:, 2].sum()
        npos_3 = target_img[:, 3].sum()

        loss_pxl_0 = loss_pxl_0.sum() / npos_0 if npos_0.item() != 0 else loss_pxl_0.sum()
        loss_pxl_1 = loss_pxl_1.sum() / npos_1 if npos_1.item() != 0 else loss_pxl_1.sum()
        # loss_pxl_2 = loss_pxl_2.sum() / npos_2 if npos_2.item() != 0 else loss_pxl_2.sum()
        loss_pxl_2 = loss_pxl_2.mean()
        loss_pxl_3 = loss_pxl_3.sum() / npos_3 if npos_3.item() != 0 else loss_pxl_3.sum()

        return loss_pxl_0, loss_pxl_1, loss_pxl_2, loss_pxl_3, loss_img


class SeverstalClsSegLossV4(nn.Module):
    def __init__(self, label_weight=None):
        super(SeverstalClsSegLossV4, self).__init__()
        self.criterion_img = lambda x,y: nn.BCEWithLogitsLoss()(x, y) + floss(x, y)
        self.criterion_pxl = MultiLabelSymmetricLovaszPerImg(label_weight=label_weight)

    def forward(self, logit_pxl, logit_img, target_pxl, target_img):
        loss_img = self.criterion_img(logit_img, target_img)

        loss_pxl_0, loss_pxl_1, loss_pxl_2, loss_pxl_3 = self.criterion_pxl(logit_pxl, target_pxl)

        loss_pxl_0 = loss_pxl_0 * target_img[:, 0:1]
        loss_pxl_1 = loss_pxl_1 * target_img[:, 1:2]
        # loss_pxl_2 = loss_pxl_2 * target_img[:, 2:3]
        loss_pxl_3 = loss_pxl_3 * target_img[:, 3:4]

        npos_0 = target_img[:, 0].sum()
        npos_1 = target_img[:, 1].sum()
        # npos_2 = target_img[:, 2].sum()
        npos_3 = target_img[:, 3].sum()

        loss_pxl_0 = loss_pxl_0.sum() / npos_0 if npos_0.item() != 0 else loss_pxl_0.sum()
        loss_pxl_1 = loss_pxl_1.sum() / npos_1 if npos_1.item() != 0 else loss_pxl_1.sum()
        # loss_pxl_2 = loss_pxl_2.sum() / npos_2 if npos_2.item() != 0 else loss_pxl_2.sum()
        loss_pxl_2 = loss_pxl_2.mean()
        loss_pxl_3 = loss_pxl_3.sum() / npos_3 if npos_3.item() != 0 else loss_pxl_3.sum()

        return loss_pxl_0, loss_pxl_1, loss_pxl_2, loss_pxl_3, loss_img


class LovaszHingeOCR(nn.Module):
    def __init__(self, label_weight=None): 
        super(LovaszHingeOCR, self).__init__()
        self.label_weight = label_weight

    def forward(self, score, target):

        assert score.size(1) == target.size(1)

        ph, pw = score.size(2), score.size(3)
        _, c, h, w = target.size()
        if ph != h or pw != w:
            score = F.interpolate(input=score, size=(h, w), mode='bilinear', align_corners=False)

        loss = 0
        for i in range(c):
            cls_loss = lovasz_hinge(score[:, i, :, :], target[:, i, :, :])
            if self.label_weight:
                cls_loss *= self.label_weight[i]
            loss += cls_loss

        return loss


def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1: # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def iou_binary(preds, labels, EMPTY=1., ignore=None, per_image=True):
    """
    IoU for foreground class
    binary: 1 foreground, 0 background
    """
    if not per_image:
        preds, labels = (preds,), (labels,)
    ious = []
    for pred, label in zip(preds, labels):
        intersection = ((label == 1) & (pred == 1)).sum()
        union = ((label == 1) | ((pred == 1) & (label != ignore))).sum()
        if not union:
            iou = EMPTY
        else:
            iou = float(intersection) / float(union)
        ious.append(iou)
    iou = mean(ious)    # mean accross images if per_image
    return 100 * iou


def iou(preds, labels, C, EMPTY=1., ignore=None, per_image=False):
    """
    Array of IoU for each (non ignored) class
    """
    if not per_image:
        preds, labels = (preds,), (labels,)
    ious = []
    for pred, label in zip(preds, labels):
        iou = []    
        for i in range(C):
            if i != ignore: # The ignored label is sometimes among predicted classes (ENet - CityScapes)
                intersection = ((label == i) & (pred == i)).sum()
                union = ((label == i) | ((pred == i) & (label != ignore))).sum()
                if not union:
                    iou.append(EMPTY)
                else:
                    iou.append(float(intersection) / float(union))
        ious.append(iou)
    ious = [mean(iou) for iou in zip(*ious)] # mean accross images if per_image
    return 100 * np.array(ious)


# --------------------------- BINARY LOSSES ---------------------------

def lovasz_per_img(logits, labels, ignore=None):
    """
    Binary Lovasz hinge loss
      logits: [B, H, W] Variable, logits at each pixel (between -\infty and +\infty)
      labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
      per_image: compute the loss per image instead of per batch
      ignore: void class id
    """
    loss = torch.empty(logits.size()[0], 1, device='cuda') if torch.cuda.is_available() else torch.empty(logits.size()[0], 1).cpu()
    for i, (log, lab) in enumerate(zip(logits, labels)):
        loss[i] = lovasz_hinge_flat(*flatten_binary_scores(log.unsqueeze(0), lab.unsqueeze(0), ignore))

    return loss


def lovasz_hinge(logits, labels, per_image=True, ignore=None):
    """
    Binary Lovasz hinge loss
      logits: [B, H, W] Variable, logits at each pixel (between -\infty and +\infty)
      labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
      per_image: compute the loss per image instead of per batch
      ignore: void class id
    """
    if per_image:
        loss = mean(lovasz_hinge_flat(*flatten_binary_scores(log.unsqueeze(0), lab.unsqueeze(0), ignore))
                          for log, lab in zip(logits, labels))
    else:
        loss = lovasz_hinge_flat(*flatten_binary_scores(logits, labels, ignore))
    return loss


def lovasz_hinge_flat(logits, labels):
    """
    Binary Lovasz hinge loss
      logits: [P] Variable, logits at each prediction (between -\infty and +\infty)
      labels: [P] Tensor, binary ground truth labels (0 or 1)
      ignore: label to ignore
    """
    if len(labels) == 0:
        # only void pixels, the gradients should be 0
        return logits.sum() * 0.
    signs = 2. * labels.float() - 1.
    errors = (1. - logits * Variable(signs))
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    perm = perm.data
    gt_sorted = labels[perm]
    grad = lovasz_grad(gt_sorted)
    # loss = torch.dot(F.relu(errors_sorted), Variable(grad))
    loss = torch.dot(F.elu(errors_sorted) + 1, Variable(grad))
    return loss


def flatten_binary_scores(scores, labels, ignore=None):
    """
    Flattens predictions in the batch (binary case)
    Remove labels equal to 'ignore'
    """
    scores = scores.view(-1)
    labels = labels.view(-1)
    if ignore is None:
        return scores, labels
    valid = (labels != ignore)
    vscores = scores[valid]
    vlabels = labels[valid]
    return vscores, vlabels


class StableBCELoss(torch.nn.modules.Module):
    def __init__(self):
         super(StableBCELoss, self).__init__()
    def forward(self, input, target):
         neg_abs = - input.abs()
         loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()
         return loss.mean()


def binary_xloss(logits, labels, ignore=None):
    """
    Binary Cross entropy loss
      logits: [B, H, W] Variable, logits at each pixel (between -\infty and +\infty)
      labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
      ignore: void class id
    """
    logits, labels = flatten_binary_scores(logits, labels, ignore)
    loss = StableBCELoss()(logits, Variable(labels.float()))
    return loss


# --------------------------- MULTICLASS LOSSES ---------------------------


def lovasz_softmax(probas, labels, classes='present', per_image=False, ignore=None):
    """
    Multi-class Lovasz-Softmax loss
      probas: [B, C, H, W] Variable, class probabilities at each prediction (between 0 and 1).
              Interpreted as binary (sigmoid) output with outputs of size [B, H, W].
      labels: [B, H, W] Tensor, ground truth labels (between 0 and C - 1)
      classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
      per_image: compute the loss per image instead of per batch
      ignore: void class labels
    """
    if per_image:
        loss = mean(lovasz_softmax_flat(*flatten_probas(prob.unsqueeze(0), lab.unsqueeze(0), ignore), classes=classes)
                          for prob, lab in zip(probas, labels))
    else:
        loss = lovasz_softmax_flat(*flatten_probas(probas, labels, ignore), classes=classes)
    return loss


def lovasz_softmax_flat(probas, labels, classes='present'):
    """
    Multi-class Lovasz-Softmax loss
      probas: [P, C] Variable, class probabilities at each prediction (between 0 and 1)
      labels: [P] Tensor, ground truth labels (between 0 and C - 1)
      classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
    """
    if probas.numel() == 0:
        # only void pixels, the gradients should be 0
        return probas * 0.
    C = probas.size(1)
    losses = []
    class_to_sum = list(range(C)) if classes in ['all', 'present'] else classes
    for c in class_to_sum:
        fg = (labels == c).float() # foreground for class c
        if (classes is 'present' and fg.sum() == 0):
            continue
        if C == 1:
            if len(classes) > 1:
                raise ValueError('Sigmoid output possible only with 1 class')
            class_pred = probas[:, 0]
        else:
            class_pred = probas[:, c]
        errors = (Variable(fg) - class_pred).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        perm = perm.data
        fg_sorted = fg[perm]
        losses.append(torch.dot(errors_sorted, Variable(lovasz_grad(fg_sorted))))
    return mean(losses)


def flatten_probas(probas, labels, ignore=None):
    """
    Flattens predictions in the batch
    """
    if probas.dim() == 3:
        # assumes output of a sigmoid layer
        B, H, W = probas.size()
        probas = probas.view(B, 1, H, W)
    B, C, H, W = probas.size()
    probas = probas.permute(0, 2, 3, 1).contiguous().view(-1, C)  # B * H * W, C = P, C
    labels = labels.view(-1)
    if ignore is None:
        return probas, labels
    valid = (labels != ignore)
    vprobas = probas[valid.nonzero().squeeze()]
    vlabels = labels[valid]
    return vprobas, vlabels

def xloss(logits, labels, ignore=None):
    """
    Cross entropy loss
    """
    return F.cross_entropy(logits, Variable(labels), ignore_index=255)


# --------------------------- HELPER FUNCTIONS ---------------------------
def isnan(x):
    return x != x
    
    
def mean(l, ignore_nan=False, empty=0):
    """
    nanmean compatible with generators.
    """
    l = iter(l)
    if ignore_nan:
        l = ifilterfalse(isnan, l)
    try:
        n = 1
        acc = next(l)
    except StopIteration:
        if empty == 'raise':
            raise ValueError('Empty mean')
        return empty
    for n, v in enumerate(l, 2):
        acc += v
    if n == 1:
        return acc
    return acc / n


# --------------------------- SURFACE LOSSES ---------------------------
class GeneralizedDice():

    def __call__(self, pc, tc):
        assert simplex(pc) and simplex(tc)

        w = 1 / ((einsum("bcwh->bc", tc).type(torch.float32) + 1e-10) ** 2)
        intersection = w * einsum("bcwh,bcwh->bc", pc, tc)
        union = w * (einsum("bcwh->bc", pc) + einsum("bcwh->bc", tc))

        divided = 1 - 2 * (einsum("bc->b", intersection) + 1e-10) / (einsum("bc->b", union) + 1e-10)

        loss = divided.mean()

        return loss


class DiceLoss():

    def __call__(self, pc, tc):
        assert simplex(pc) and simplex(tc)

        intersection = einsum("bcwh,bcwh->bc", pc, tc)
        union = (einsum("bcwh->bc", pc) + einsum("bcwh->bc", tc))

        divided = 1 - (2 * intersection + 1e-10) / (union + 1e-10)

        loss = divided.mean()

        return loss


class SurfaceLoss():

    def __call__(self, pc, dc):
        assert simplex(pc)
        assert not one_hot(dc)

        multipled = einsum("bcwh,bcwh->bcwh", pc, dc)

        loss = multipled.mean()

        return loss


def generalized_dice(score, target):
    ph, pw = score.size(2), score.size(3)
    h, w = target.size(1), target.size(2)
    if ph != h or pw != w:
        score = F.interpolate(input=score, size=(h, w), mode='bilinear', align_corners=False)

    pred = F.softmax(score, dim=1)
    target = make_one_hot_multiclass(target, 5)
    gdl = GeneralizedDice()
    return gdl(pred, target)


def surface_loss(score, target, dist_map):
    ph, pw = score.size(2), score.size(3)
    h, w = target.size(1), target.size(2)
    if ph != h or pw != w:
        score = F.interpolate(input=score, size=(h, w), mode='bilinear', align_corners=False)

    pred = F.softmax(score, dim=1)
    segloss = OhemCrossEntropy()
    surface = SurfaceLoss()
    return segloss(score, target), surface(pred, dist_map)


def alpha_scheduler(epoch):
    alpha_arr = np.arange(1, 0, -0.01)
    if epoch < len(alpha_arr):
        return alpha_arr[epoch]
    else:
        return 0.01
