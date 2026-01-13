import torch.nn as nn

class VisionNet(nn.Module):
    """(predicting a vision quality score)."""
    def __init__(self, in_dim: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(64,32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)
