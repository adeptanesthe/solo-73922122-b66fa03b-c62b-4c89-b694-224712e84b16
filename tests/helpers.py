"""Shared test helpers: synthetic 16 kHz PCM16 material."""

from __future__ import annotations

import numpy as np

from app.constants import FRAME_SAMPLES


def tone(period: int, samples: int, amplitude: float = 12_000.0) -> np.ndarray:
    t = np.arange(samples)
    wave = amplitude * np.sin(2 * np.pi * t / period)
    return np.clip(np.rint(wave), -32768, 32767).astype(np.int16)


def tone_frames(count: int, period: int = 160) -> list[np.ndarray]:
    """One continuous tone split into 20 ms frames (phase-continuous)."""

    full = tone(period, FRAME_SAMPLES * count)
    return [
        full[i * FRAME_SAMPLES : (i + 1) * FRAME_SAMPLES].copy()
        for i in range(count)
    ]
