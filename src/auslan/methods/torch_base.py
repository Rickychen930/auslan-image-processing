"""Shared PyTorch training loop for M3, M4 and the M5 MLP.

Subclasses say how to build the network, which parameters form the classification head and how
to turn samples into batches. Training, early stopping and ASL -> Auslan weight transfer
(protocol P3) live here once.
"""

from __future__ import annotations

import copy
import logging
import math
import random
from typing import Self

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset

from auslan.data.sample_set import SampleSet
from auslan.evaluation.metrics import macro_f1
from auslan.methods.base import Classifier

log = logging.getLogger(__name__)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
MIN_EVAL_EVERY = 25  # validate at least this many steps apart, so early stopping is not twitchy
MAX_EVAL_EVERY = 250
ROTATION_DEGREES = 10
CROP_SIDE = (0.9, 1.0)  # fraction of the image side kept by the random crop
JITTER = 0.2  # brightness, contrast and saturation vary by up to this fraction


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class TorchClassifier:
    """Mini-batch trainer with a step budget, cosine LR, early stopping and head-less transfer."""

    supports_transfer = True
    uses_landmarks = False
    needs_colour = False
    head_prefixes: tuple[str, ...] = ("head",)  # parameters that are re-initialised for new classes

    def __init__(
        self,
        name: str,
        *,
        max_steps: int = 1000,
        max_epochs: int = 30,
        min_steps: int = 300,
        batch_size: int = 64,
        lr: float = 3e-4,
        weight_decay: float = 1e-4,
        patience: int = 5,
        device: str | None = None,
        num_workers: int = 0,
        seed: int = 0,
    ) -> None:
        self.name = name
        self._max_steps, self._max_epochs, self._min_steps = max_steps, max_epochs, min_steps
        self._batch_size, self._lr, self._weight_decay = batch_size, lr, weight_decay
        self._patience, self._num_workers, self._seed = patience, num_workers, seed
        self._device = device or default_device()
        self._model: nn.Module | None = None

    def _build(self, n_classes: int) -> nn.Module:
        raise NotImplementedError

    def _dataset(self, samples: SampleSet, train: bool) -> Dataset:
        raise NotImplementedError

    def _prepare(self, batch: torch.Tensor, train: bool) -> torch.Tensor:
        """Turn a loaded batch (already on the device) into network input."""
        return batch

    def fit(
        self, train: SampleSet, val: SampleSet | None = None, init: Classifier | None = None
    ) -> Self:
        random.seed(self._seed)
        np.random.seed(self._seed)  # noqa: NPY002 - augmentation libraries read the global state
        torch.manual_seed(self._seed)
        if len(train) == 0:
            raise ValueError(f"{self.name}: the training set is empty")
        n_classes = len(train.label_space)
        loader = self._loader(train, shuffle=True)
        val_loader = self._loader(val, shuffle=False) if val is not None else None
        model = self._build(n_classes)
        if init is not None:
            self._transfer(model, init)
        model.to(self._device)
        self._model = self._train(model, loader, val_loader, val, n_classes)
        return self

    def predict_proba(self, samples: SampleSet) -> np.ndarray:
        if self._model is None:
            raise RuntimeError(f"{self.name} must be fitted before predicting")
        logits = self._logits(self._model, self._loader(samples, shuffle=False))
        return F.softmax(logits, dim=1).numpy().astype(np.float32)

    def use_device(self, device: str) -> None:
        """Move the fitted model, e.g. to the CPU when timing laptop inference."""
        self._device = device
        if self._model is not None:
            self._model.to(device)

    def _loader(self, samples: SampleSet, shuffle: bool) -> DataLoader:
        dataset = self._dataset(samples, train=shuffle)
        return DataLoader(
            dataset,
            batch_size=self._batch_size,
            shuffle=shuffle,
            drop_last=shuffle and len(dataset) % self._batch_size == 1,  # BatchNorm needs > 1
            num_workers=self._num_workers,
        )

    def _transfer(self, model: nn.Module, init: Classifier) -> None:
        """Copy every weight except the classification head from a model trained on ASL."""
        if not isinstance(init, type(self)) or init._model is None:
            raise TypeError(f"{self.name} can only start from a fitted {type(self).__name__}")
        state = init._model.state_dict()
        weights = {k: v for k, v in state.items() if not k.startswith(self.head_prefixes)}
        if len(weights) == len(state):
            raise RuntimeError(
                f"head_prefixes {self.head_prefixes} match no parameter of {self.name}"
            )
        _, unexpected = model.load_state_dict(weights, strict=False)
        if unexpected:
            raise RuntimeError(f"unexpected keys while transferring weights: {unexpected}")

    def _train(
        self,
        model: nn.Module,
        loader: DataLoader,
        val_loader: DataLoader | None,
        val: SampleSet | None,
        n_classes: int,
    ) -> nn.Module:
        # Tiny few-shot sets get at least min_steps; huge sets are cut off at max_steps.
        budget = min(self._max_steps, max(self._max_epochs * len(loader), self._min_steps))
        trainable = [p for p in model.parameters() if p.requires_grad]
        optimiser = AdamW(trainable, lr=self._lr, weight_decay=self._weight_decay)
        scheduler = CosineAnnealingLR(optimiser, T_max=budget)
        eval_every = min(max(len(loader), MIN_EVAL_EVERY), MAX_EVAL_EVERY)
        best_score, best_state, stale, step = -1.0, None, 0, 0
        while step < budget and stale < self._patience:
            for x, y in loader:
                model.train()
                inputs = self._prepare(x.to(self._device), train=True)
                loss = F.cross_entropy(model(inputs), y.to(self._device))
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()
                scheduler.step()
                step += 1
                if val_loader is not None and (step % eval_every == 0 or step == budget):
                    score = self._val_score(model, val_loader, val, n_classes)
                    if score > best_score:
                        best_score, best_state, stale = score, copy.deepcopy(model.state_dict()), 0
                    else:
                        stale += 1
                if step >= budget or stale >= self._patience:
                    break
        log.info(
            "%s: %d steps (%.1f epochs), best val macro-F1 %.3f",
            self.name,
            step,
            step / len(loader),
            best_score,
        )
        if best_state is not None:
            model.load_state_dict(best_state)
        return model

    def _val_score(self, model: nn.Module, loader: DataLoader, val: SampleSet, n: int) -> float:
        predictions = self._logits(model, loader).argmax(dim=1).numpy()
        return macro_f1(val.labels, predictions, n)

    def _logits(self, model: nn.Module, loader: DataLoader) -> torch.Tensor:
        model.eval()
        outputs = []
        with torch.no_grad():
            for x, _ in loader:
                outputs.append(model(self._prepare(x.to(self._device), train=False)).float().cpu())
        return torch.cat(outputs)


class _ImageDataset(Dataset):
    """Decoded crops as uint8 tensors; augmentation happens later, on the whole batch."""

    def __init__(self, samples: SampleSet) -> None:
        self._samples = samples

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int):
        crop = torch.from_numpy(self._samples[index].crop).permute(2, 0, 1)
        return crop, int(self._samples.labels[index])


def _uniform(n: int, low: float, high: float, device: torch.device, *shape: int) -> torch.Tensor:
    return low + (high - low) * torch.rand(n, *shape, device=device)


def augment_batch(images: torch.Tensor, fill: float) -> torch.Tensor:
    """Random rotation, crop and shift plus brightness, contrast and saturation jitter.

    ``images`` is float in [0, 1]. There is no flip: it would swap the two hands. Corners that
    the rotation exposes are filled with ``fill`` (the page colour), not with ink.
    """
    n, device = len(images), images.device
    angle = _uniform(n, -1, 1, device) * math.radians(ROTATION_DEGREES)
    side = _uniform(n, *CROP_SIDE, device)
    shift = _uniform(n, -1, 1, device, 2) * (1 - side)[:, None]  # keeps the crop inside the image
    theta = torch.stack(
        [
            torch.stack([side * angle.cos(), -side * angle.sin(), shift[:, 0]], dim=1),
            torch.stack([side * angle.sin(), side * angle.cos(), shift[:, 1]], dim=1),
        ],
        dim=1,
    )
    grid = F.affine_grid(theta, list(images.shape), align_corners=False)
    out = F.grid_sample(images - fill, grid, padding_mode="zeros", align_corners=False) + fill

    jitter = [_uniform(n, 1 - JITTER, 1 + JITTER, device)[:, None, None, None] for _ in range(3)]
    brightness, contrast, saturation = jitter
    out = out * brightness
    mean = out.mean(dim=(1, 2, 3), keepdim=True)
    out = (out - mean) * contrast + mean
    grey = out.mean(dim=1, keepdim=True)
    return ((out - grey) * saturation + grey).clamp(0, 1)


class ImageClassifier(TorchClassifier):
    """Image models with ImageNet normalisation and (optionally) batch augmentation."""

    def __init__(self, name: str, *, augment: bool = True, **options: object) -> None:
        super().__init__(name, **options)
        self._augment = augment
        self._fill = 0.0
        self._mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
        self._std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)

    def fit(
        self, train: SampleSet, val: SampleSet | None = None, init: Classifier | None = None
    ) -> Self:
        self._fill = 1.0 if train.mostly_binarised else 0.0  # white pages stay white when rotated
        return super().fit(train, val, init)

    def _dataset(self, samples: SampleSet, train: bool) -> Dataset:
        return _ImageDataset(samples)

    def _prepare(self, batch: torch.Tensor, train: bool) -> torch.Tensor:
        images = batch.float() / 255
        if train and self._augment:
            images = augment_batch(images, self._fill)
        return (images - self._mean.to(images.device)) / self._std.to(images.device)
