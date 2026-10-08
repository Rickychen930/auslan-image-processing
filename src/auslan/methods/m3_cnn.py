"""M3: a four-block CNN trained from random weights (no ImageNet pretraining)."""

from __future__ import annotations

from torch import nn

from auslan.methods.base import register
from auslan.methods.torch_base import ImageClassifier

CHANNELS = (32, 64, 128, 256)
DROPOUT = 0.3


def _block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class SmallCnn(nn.Module):
    def __init__(self, n_classes: int) -> None:
        super().__init__()
        widths = (3, *CHANNELS)
        self.features = nn.Sequential(
            *(_block(a, b) for a, b in zip(widths, widths[1:], strict=False))
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(), nn.Dropout(DROPOUT), nn.Linear(CHANNELS[-1], n_classes)
        )

    def forward(self, x):
        return self.head(self.pool(self.features(x)))


class CnnClassifier(ImageClassifier):
    head_prefixes = ("head",)

    def _build(self, n_classes: int) -> nn.Module:
        return SmallCnn(n_classes)


@register("m3_cnn", seeded=True)
def _cnn(**options: object) -> CnnClassifier:
    return CnnClassifier("m3_cnn", **options)
