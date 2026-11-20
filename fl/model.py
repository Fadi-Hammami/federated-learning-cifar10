"""
Global model: a small CNN for CIFAR-10 (3x32x32 colour images, 10 classes).

    Conv(3->32, 3x3, pad 1) + ReLU + MaxPool(2)   -> 32 x 16 x 16
    Conv(32->64, 3x3, pad 1) + ReLU + MaxPool(2)  -> 64 x 8 x 8
    Conv(64->128, 3x3, pad 1) + ReLU + MaxPool(2) -> 128 x 4 x 4
    Flatten (2048) -> FC 256 + ReLU -> Dropout(0.3) -> FC 10

~ 620k trainable parameters. Deeper than FedLab's CNN_CIFAR10 (LeNet-style, 62k params),
which plateaus around 55 % on CIFAR-10; this one reaches ~70 % centralised in a few epochs
while staying fast enough on CPU.
"""
import torch.nn as nn


class CNN_CIFAR10(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(128 * 4 * 4, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, 10),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
