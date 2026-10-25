import torch.nn as nn


class SimpleConvEncoder(nn.Module):
    def __init__(self, in_channels: int, embed_dim: int = 128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(embed_dim, in_channels, kernel_size=1),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.encoder(x) + x
