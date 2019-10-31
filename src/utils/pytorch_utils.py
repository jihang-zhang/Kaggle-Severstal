import torch

def make_one_hot_binary(labels, num_classes):
    labels.unsqueeze_(1)
    one_hot = torch.cuda.FloatTensor(labels.size(0), num_classes, labels.size(2), labels.size(3)).zero_()
    target = one_hot.scatter_(1, labels.data, 1)

    return target[:, 1:, :, :]

def make_one_hot_multiclass(labels, num_classes):
    labels.unsqueeze_(1)
    one_hot = torch.cuda.FloatTensor(labels.size(0), num_classes, labels.size(2), labels.size(3)).zero_()
    target = one_hot.scatter_(1, labels.data, 1)

    return target