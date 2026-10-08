"""Letter vocabularies shared by every dataset and method."""

from __future__ import annotations

import string
from dataclasses import dataclass

LETTERS: tuple[str, ...] = tuple(string.ascii_uppercase)
_MOTION_LETTERS = frozenset("JZ")  # need movement; single-frame images cannot show them

ASL_LETTERS: tuple[str, ...] = tuple(c for c in LETTERS if c not in _MOTION_LETTERS)
AUSLAN_LETTERS: tuple[str, ...] = LETTERS


def letter_from_name(name: str) -> str | None:
    """Map a folder or label name to a letter, or ``None`` for digits and non-letter classes."""
    candidate = name.strip().upper()
    return candidate if len(candidate) == 1 and candidate in LETTERS else None


@dataclass(frozen=True)
class LabelSpace:
    """Ordered set of letters; the position of a letter is its integer class id."""

    letters: tuple[str, ...]

    def __len__(self) -> int:
        return len(self.letters)

    def index(self, letter: str) -> int:
        try:
            return self.letters.index(letter)
        except ValueError:
            raise KeyError(letter) from None

    def letter(self, index: int) -> str:
        return self.letters[index]


ASL_SPACE = LabelSpace(ASL_LETTERS)
AUSLAN_SPACE = LabelSpace(AUSLAN_LETTERS)
