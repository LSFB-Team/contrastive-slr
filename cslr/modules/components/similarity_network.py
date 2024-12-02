from torch import nn
import torch


class SimilarityNetwork(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def sim(self, zi, e):
        return torch.dot(zi, e) / (torch.norm(zi) * torch.norm(e))

    def forward(self, z, e):

        c = torch.stack([self.sim(zi, e) for zi in torch.unbind(z, dim=0)], dim=0)
        q = torch.softmax(c / self.temperature, dim=0)
        return q
