from torch import nn


class ClassificationHead(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: tuple[int], out_channels: int, use_batch_norm: bool):
        super().__init__()
        assert len(hidden_channels) > 0, "The classification head needs at least one hidden layer."
        layers = []
        h_in = in_channels
        for h_out in hidden_channels:
            layers += [
                nn.Linear(h_in, h_out),
                nn.BatchNorm1d(h_out) if use_batch_norm else nn.Identity(),
                nn.ReLU(),
            ]
        self.mlp = nn.Sequential(
            *layers,
            nn.Linear(hidden_channels[-1], out_channels)
        )

    def forward(self, x):
        return self.mlp(x)
