"""M4: ImageNet-pretrained MobileNetV2 and ResNet-18, fine-tuned end to end (or frozen)."""

from __future__ import annotations

from collections.abc import Callable

from torch import nn
from torchvision import models

from auslan.methods.base import register
from auslan.methods.torch_base import ImageClassifier


def _mobilenet(n_classes: int, pretrained: bool) -> nn.Module:
    weights = models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
    net = models.mobilenet_v2(weights=weights)
    net.classifier[1] = nn.Linear(net.last_channel, n_classes)
    return net


def _resnet(n_classes: int, pretrained: bool) -> nn.Module:
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    net = models.resnet18(weights=weights)
    net.fc = nn.Linear(net.fc.in_features, n_classes)
    return net


class TorchvisionClassifier(ImageClassifier):
    """A torchvision backbone with a new classification layer; can train only that layer."""

    def __init__(
        self,
        name: str,
        build: Callable[[int, bool], nn.Module],
        head_prefixes: tuple[str, ...],
        pretrained: bool = True,
        freeze_backbone: bool = False,
        **options: object,
    ) -> None:
        super().__init__(name, **options)
        self.head_prefixes = head_prefixes
        self._build_net = build
        self._pretrained = pretrained
        self._freeze = freeze_backbone

    def _build(self, n_classes: int) -> nn.Module:
        net = self._build_net(n_classes, self._pretrained)
        if self._freeze:
            for name, parameter in net.named_parameters():
                parameter.requires_grad = name.startswith(self.head_prefixes)
        return net


@register("m4_mobilenetv2", seeded=True)
def _mobilenet_classifier(**options: object) -> TorchvisionClassifier:
    return TorchvisionClassifier("m4_mobilenetv2", _mobilenet, ("classifier.1",), **options)


@register("m4_resnet18", seeded=True)
def _resnet_classifier(**options: object) -> TorchvisionClassifier:
    return TorchvisionClassifier("m4_resnet18", _resnet, ("fc",), **options)
