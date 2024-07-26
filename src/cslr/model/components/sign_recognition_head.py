from torch import nn


class SimilarityNetwork(nn.Module):
    def __init__(self, temperature=0.1):
        # TODO