"""Webcam loop: live preview, the model's top guesses and an optional correct / wrong verdict."""

from __future__ import annotations

import time

import cv2
import numpy as np

from auslan.live.recognizer import Recognizer, Verdict, judge, top_letters

WINDOW = "Auslan fingerspelling"
STRIDE = 2  # classify every second frame; the preview stays smooth
ESCAPE = 27
THUMBNAIL = 96
COLOURS = {  # BGR
    "idle": (235, 235, 235),
    "no-hand": (0, 165, 255),
    "correct": (60, 200, 60),
    "wrong": (60, 60, 230),
}
HINT = "letter key: set target   space: clear target   esc: quit"


def draw_overlay(
    preview: np.ndarray,
    guesses: list[tuple[str, float]],
    verdict: Verdict,
    fps: float,
    model_input: np.ndarray | None,
) -> np.ndarray:
    """Draw the verdict, the top guesses, the model's input and the frame rate on a copy."""
    image = preview.copy()
    height, width = image.shape[:2]
    colour = COLOURS[verdict.status]
    cv2.rectangle(image, (0, 0), (width, 44), (0, 0, 0), -1)
    cv2.putText(image, verdict.message, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, colour, 2)
    for row, (letter, probability) in enumerate(guesses):
        y = 70 + 34 * row
        cv2.putText(image, letter, (12, y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.rectangle(image, (48, y), (48 + int(160 * probability), y + 22), colour, -1)
        cv2.putText(
            image,
            f"{probability:.0%}",
            (216, y + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
        )
    if model_input is not None:
        thumb = cv2.resize(model_input, (THUMBNAIL, THUMBNAIL), interpolation=cv2.INTER_AREA)
        image[52 : 52 + THUMBNAIL, width - THUMBNAIL - 8 : width - 8] = cv2.cvtColor(
            thumb, cv2.COLOR_RGB2BGR
        )
        cv2.putText(
            image,
            "model input",
            (width - THUMBNAIL - 8, 52 + THUMBNAIL + 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
        )
    cv2.putText(
        image,
        f"{fps:.0f} fps",
        (width - 90, height - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
    )
    cv2.putText(image, HINT, (12, height - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return image


def run_camera(
    recognizer: Recognizer, letters: tuple[str, ...], target: str | None = None, camera: int = 0
) -> None:
    """Show the camera until Escape is pressed or the window is closed."""
    capture = cv2.VideoCapture(camera)
    if not capture.isOpened():
        raise RuntimeError(
            f"cannot open camera {camera}; on macOS allow your terminal under "
            "System Settings > Privacy & Security > Camera"
        )
    proba, frame_number, fps, previous = None, 0, 0.0, time.perf_counter()
    try:
        while True:
            ok, bgr = capture.read()
            if not ok:
                raise RuntimeError("the camera stopped delivering frames")
            if frame_number % STRIDE == 0:
                proba = recognizer.proba(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))  # not mirrored
            frame_number += 1
            now = time.perf_counter()
            fps = 0.9 * fps + 0.1 / max(now - previous, 1e-6)
            previous = now

            guesses = top_letters(letters, proba) if proba is not None else []
            verdict = judge(letters, proba, target)
            shown = draw_overlay(cv2.flip(bgr, 1), guesses, verdict, fps, recognizer.last_input)
            cv2.imshow(WINDOW, shown)

            key = cv2.waitKey(1) & 0xFF
            if key == ESCAPE or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key == ord(" "):
                target = None
            elif chr(key).upper() in letters:
                target = chr(key).upper()
    finally:
        capture.release()
        cv2.destroyAllWindows()
