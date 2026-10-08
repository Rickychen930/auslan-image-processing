"""M4: ImageNet-pretrained MobileNetV2 and ResNet-18, fine-tuned end to end."""

from __future__ import annotations

from torch import nn
from torchvision import models

from auslan.methods.base import register
from auslan.methods.torch_base import ImageClassifier


class MobileNetV2Classifier(ImageClassifier):
    head_prefixes = ("classifier.1",)

    def __init__(self, name: str, pretrained: bool = True, **options: object) -> None:
        super().__init__(name, **options)
        self._pretrained = pretrained

    def _build(self, n_classes: int) -> nn.Module:
        weights = models.MobileNet_V2_Weights.IMAGENET1K_V1 if self._pretrained else None
        net = models.mobilenet_v2(weights=weights)
        net.classifier[1] = nn.Linear(net.last_channel, n_classes)
        return net


class ResNet18Classifier(ImageClassifier):
    head_prefixes = ("fc",)

    def __init__(self, name: str, pretrained: bool = True, **options: object) -> None:
        super().__init__(name, **options)
        self._pretrained = pretrained

    def _build(self, n_classes: int) -> nn.Module:
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if self._pretrained else None
        net = models.resnet18(weights=weights)
        net.fc = nn.Linear(net.fc.in_features, n_classes)
        return net


@register("m4_mobilenetv2", seeded=True)
def _mobilenet(**options: object) -> MobileNetV2Classifier:
    return MobileNetV2Classifier("m4_mobilenetv2", **options)


@register("m4_resnet18", seeded=True)
def _resnet(**options: object) -> ResNet18Classifier:
    return ResNet18Classifier("m4_resnet18", **options)
