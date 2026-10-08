"""Shared PyTorch training loop for M3, M4 and the M5 MLP.

Subclasses say how to build the network, which parameters form the classification head and how
to turn samples into batches. Training, early stopping and ASL -> Auslan weight transfer
(protocol P3) live here once.
"""

from __future__ import annotations

import copy
import logging
import random
from typing import Self

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

from auslan.data.sample_set import SampleSet
from auslan.evaluation.metrics import macro_f1
from auslan.methods.base import Classifier

log = logging.getLogger(__name__)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
MIN_EVAL_EVERY = 25  # validate at least this many steps apart, so early stopping is not twitchy
MAX_EVAL_EVERY = 250
WHITE, BLACK = 255, 0


def default_device() -> str:
    return "mps" if torch.backends.mps.is_available() else "cpu"


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
        max_steps: int = 1500,
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
                loss = F.cross_entropy(model(x.to(self._device)), y.to(self._device))
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
                outputs.append(model(x.to(self._device)).float().cpu())
        return torch.cat(outputs)


class _ImageDataset(Dataset):
    def __init__(self, samples: SampleSet, transform: v2.Compose) -> None:
        self._samples, self._transform = samples, transform

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int):
        return self._transform(self._samples[index].crop), int(self._samples.labels[index])


class ImageClassifier(TorchClassifier):
    """Image models. Augmentation never flips: a flip would swap the two hands."""

    def __init__(self, name: str, *, augment: bool = True, **options: object) -> None:
        super().__init__(name, **options)
        self._augment = augment

    def _dataset(self, samples: SampleSet, train: bool) -> Dataset:
        steps: list = [v2.ToImage()]
        if train and self._augment:
            # Rotation exposes corners: fill them with the page colour, not with ink.
            fill = WHITE if samples.mostly_binarised else BLACK
            steps += [
                v2.RandomResizedCrop(
                    samples.image_size, scale=(0.8, 1.0), ratio=(0.9, 1.1), antialias=True
                ),
                v2.RandomRotation(10, fill=fill),
                v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
            ]
        steps += [v2.ToDtype(torch.float32, scale=True), v2.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
        return _ImageDataset(samples, v2.Compose(steps))
