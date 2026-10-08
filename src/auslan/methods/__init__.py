"""Importing the method modules registers every method with ``auslan.methods.base``."""

from auslan.methods import m1_skin, m2_hog, m3_cnn, m4_pretrained, m5_landmarks  # noqa: F401
from auslan.methods.base import Classifier, create, method_names

__all__ = ["Classifier", "create", "method_names"]
