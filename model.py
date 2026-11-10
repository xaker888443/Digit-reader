"""CNN-классификатор цифр 0-9."""
from __future__ import annotations

import torch
from torch import Tensor, nn


class DigitCNN(nn.Module):
    """Две свёртки + два полносвязных слоя.

    Вход:  (N, 1, 28, 28)
    Выход: (N, 10) — логиты.
    """

    _FEATURE_MAP = 32 * 7 * 7

    def __init__(self, n_classes: int = 10) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(0.3)
        self.fc1 = nn.Linear(self._FEATURE_MAP, 128)
        self.fc2 = nn.Linear(128, n_classes)

    def forward(self, batch: Tensor) -> Tensor:
        h = self.pool(self.act(self.conv1(batch)))   # (N,16,14,14)
        h = self.pool(self.act(self.conv2(h)))       # (N,32,7,7)
        h = h.flatten(1)
        h = self.drop(self.act(self.fc1(h)))
        return self.fc2(h)


if __name__ == "__main__":
    net = DigitCNN()
    sample = torch.zeros(1, 1, 28, 28)
    print("out:", net(sample).shape)
    print("params:", sum(p.numel() for p in net.parameters()))