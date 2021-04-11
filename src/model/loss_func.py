import math
import torch
import torch.nn as nn


class CosineMarginCrossEntropy(nn.Module):
    """
    Reference : https://github.com/melgor/kaggle-whale-tail/blob/6dc04e3fbca4dba271369cf412acf7da4ef69c3a/models/layers.py
    """  # noqa

    def __init__(self, m=0.60, s=30.0):
        super(CosineMarginCrossEntropy, self).__init__()
        self.m = m
        self.s = s
        self.ce = nn.CrossEntropyLoss()

    def forward(self, input, target):
        one_hot = torch.zeros_like(input)
        one_hot.scatter_(1, target.view(-1, 1), 1.0)
        output = self.s * (input - one_hot * self.m)

        loss = self.ce(output, target)
        return loss


class ArcMarginCrossEntropy(nn.Module):
    """
    Reference : https://github.com/melgor/kaggle-whale-tail/blob/6dc04e3fbca4dba271369cf412acf7da4ef69c3a/models/layers.py
    """  # noqa

    def __init__(self, m=0.50, s=30.0, m_cos=0.3):
        super(ArcMarginCrossEntropy, self).__init__()
        self.m = m
        self.m_cos = m_cos
        self.s = s
        self.ce = nn.CrossEntropyLoss()

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, cosine, target):

        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # convert label to one-hot
        one_hot = torch.zeros(cosine.size(), device='cuda')
        one_hot.scatter_(1, target.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s

        loss = self.ce(output, target)
        return loss
