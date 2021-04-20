import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# class CosFace(nn.Module):
#     def __init__(self, s=64.0, m=0.40):
#         super(CosFace, self).__init__()
#         self.s = s
#         self.m = m

#     def forward(self, cosine, label):
#         index = torch.where(label != -1)[0]
#         m_hot = torch.zeros(index.size()[0], cosine.size()[
#                             1], device=cosine.device)
#         m_hot = m_hot.scatter(1, label[index, None], self.m)
#         cosine[index] -= m_hot
#         ret = cosine * self.s
#         return ret


# class ArcFace(nn.Module):
#     def __init__(self, s=64.0, m=0.5):
#         super(ArcFace, self).__init__()
#         self.s = s
#         self.m = m

#     def forward(self, cosine: torch.Tensor, label):
#         index = torch.where(label != -1)[0]
#         m_hot = torch.zeros(index.size()[0], cosine.size()[
#                             1], device=cosine.device)
#         m_hot = m_hot.scatter(1, label[index, None], self.m)
#         cosine = cosine.acos()
#         cosine[index] += m_hot
#         cosine = cosine.cos()
#         cosine = cosine.mul(self.s)
#         return cosine


class ArcMarginProduct(nn.Module):
    """
    Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin
            cos(theta + m)
    Reference: https://github.com/ronghuaiyang/arcface-pytorch/blob/47ace80b128042cd8d2efd408f55c5a3e156b032/models/metrics.py#L10
    """  # noqa

    def __init__(self,
                 in_features,
                 out_features,
                 s=30.0, m=0.50,
                 easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(
            torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, data, label):

        # cos(theta) & phi(theta)
        cosine = F.linear(F.normalize(data), F.normalize(self.weight))
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # convert label to one-hot
        one_hot = torch.zeros(cosine.size(), device=data.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)

        # torch.where(out_i = {x_i if condition_i else y_i)
        # you can use torch.where if your torch.__version__ is 0.4
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s

        return output
