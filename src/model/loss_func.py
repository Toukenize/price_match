import torch
import torch.nn as nn


class CosFace(nn.Module):
    def __init__(self, s=64.0, m=0.40):
        super(CosFace, self).__init__()
        self.s = s
        self.m = m

    def forward(self, cosine, label):
        index = torch.where(label != -1)[0]
        m_hot = torch.zeros(index.size()[0], cosine.size()[
                            1], device=cosine.device)
        m_hot = m_hot.scatter(1, label[index, None], self.m)
        cosine[index] -= m_hot
        ret = cosine * self.s
        return ret


class ArcFace(nn.Module):
    def __init__(self, s=64.0, m=0.5):
        super(ArcFace, self).__init__()
        self.s = s
        self.m = m

    def forward(self, cosine: torch.Tensor, label):
        index = torch.where(label != -1)[0]
        m_hot = torch.zeros(index.size()[0], cosine.size()[
                            1], device=cosine.device)
        m_hot = m_hot.scatter(1, label[index, None], self.m)
        cosine = cosine.acos()
        cosine[index] += m_hot
        cosine = cosine.cos()
        cosine = cosine.mul(self.s)
        return cosine
