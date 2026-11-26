"""NumPy concealment primitives.

All arrays are mono int16 samples.  The history handed to these functions is
the *played* signal (real frames plus earlier concealment), exactly as it was
emitted, which makes concealment reproducible and immune to late packets.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .constants import (
    CONCEAL_GAIN,
    FADE_SAMPLES,
    FRAME_SAMPLES,
    HISTORY_SAMPLES,
    MAX_PERIOD_SAMPLES,
    MIN_PEAK_CORRELATION,
    MIN_PERIOD_SAMPLES,
)


@dataclass(frozen=True)
class PeriodEstimate:
    """Result of the normalised autocorrelation search."""

    period_samples: int | None
    peak: float | None
    reason: str  # "periodic", "insufficient_history", "autocorr_undefined", "low_correlation"


@dataclass(frozen=True)
class Concealment:
    """Synthetic frame plus the information used to build it."""

    samples: np.ndarray  # int16, FRAME_SAMPLES long
    applied: bool  # True when a period was repeated
    reason: str
    period_samples: int | None
    peak: float | None


def _to_pcm16(values: np.ndarray) -> np.ndarray:
    """Round half-to-even and clip to the signed 16-bit range."""

    return np.clip(np.rint(values), -32768, 32767).astype(np.int16)


def estimate_period(history: np.ndarray) -> PeriodEstimate:
    """Find the strongest normalised autocorrelation lag in the last 40 ms.

    The autocorrelation is mean-removed and normalised per lag, using the
    overlapping samples only:

        r(L) = sum(x[i] * x[i+L]) / sqrt(sum(x[i]^2) * sum(x[i+L]^2))

    for i = 0 .. N-L-1.  A lag whose window has no energy is undefined and is
    skipped.  The lowest lag wins ties because the scan is ascending.
    """

    x = np.asarray(history, dtype=np.int16)
    if x.ndim != 1:
        raise ValueError("history must be one-dimensional")
    if x.size < HISTORY_SAMPLES:
        return PeriodEstimate(None, None, "insufficient_history")

    window = x[-HISTORY_SAMPLES:].astype(np.float64)
    window -= window.mean()

    lags = np.arange(MIN_PERIOD_SAMPLES, MAX_PERIOD_SAMPLES + 1)
    size = window.size

    total_energy = float(np.dot(window, window))
    cumulative = np.concatenate(([0.0], np.cumsum(window * window)))
    # Lag L correlates window[:-L] with window[L:].
    numerator = np.correlate(window, window, mode="full")[size - 1 + lags]
    energy_left = cumulative[size - lags]
    energy_right = np.maximum(total_energy - cumulative[lags], 0.0)

    defined = (energy_left > 0.0) & (energy_right > 0.0)
    if not np.any(defined):
        return PeriodEstimate(None, None, "autocorr_undefined")

    peaks = np.full(lags.size, -np.inf, dtype=np.float64)
    peaks[defined] = numerator[defined] / np.sqrt(
        energy_left[defined] * energy_right[defined]
    )
    # Rounding makes exact ties land within ~1e-15 of each other; treat that
    # band as a tie and keep the shortest lag (the fundamental).
    best_peak = float(np.max(peaks))
    tied = np.flatnonzero(peaks >= best_peak - 1e-9)
    best_lag = int(lags[int(tied[0])])
    best_peak = min(best_peak, 1.0)

    if best_peak < MIN_PEAK_CORRELATION:
        return PeriodEstimate(None, best_peak, "low_correlation")
    return PeriodEstimate(best_lag, best_peak, "periodic")


def concealment_for(history: np.ndarray) -> Concealment:
    """Build one 20 ms concealment frame from the last 40 ms of played audio.

    * Too little history, undefined autocorrelation, or a peak below 0.6
      yields silence.
    * Otherwise the last detected period is tiled over 320 samples at half
      amplitude.  Because the emitted frame becomes part of the history, a
      run of missing frames recursively repeats (and halves) the newest
      period.
    """

    x = np.asarray(history, dtype=np.int16)
    estimate = estimate_period(x)
    if estimate.reason != "periodic" or estimate.period_samples is None:
        return Concealment(
            samples=np.zeros(FRAME_SAMPLES, dtype=np.int16),
            applied=False,
            reason=estimate.reason,
            period_samples=None,
            peak=estimate.peak,
        )
    period = x[-estimate.period_samples :]
    repeats = int(np.ceil(FRAME_SAMPLES / period.size))
    tiled = np.tile(period, repeats)[:FRAME_SAMPLES]
    concealed = _to_pcm16(tiled.astype(np.float64) * CONCEAL_GAIN)
    return Concealment(
        samples=concealed,
        applied=True,
        reason="periodic",
        period_samples=estimate.period_samples,
        peak=estimate.peak,
    )


def crossfade_to_true(
    concealment: np.ndarray,
    true_frame: np.ndarray,
    fade_samples: int = FADE_SAMPLES,
) -> np.ndarray:
    """Linearly move from hypothetical concealment to real audio.

    Over the first 5 ms, alpha goes from 0 at sample 0 to 1 at sample 79, so
    the first sample is exactly the concealment and sample 79 is already the
    real sample.  Every sample from 80 onward is returned untouched.
    """

    hidden = np.asarray(concealment, dtype=np.int16)
    true = np.asarray(true_frame, dtype=np.int16)
    if hidden.shape != true.shape:
        raise ValueError("concealment and true frame must have the same shape")

    out = true.copy()
    count = min(max(fade_samples, 0), true.size)
    if count == 0:
        return out
    if count == 1:
        out[0] = hidden[0]
        return out

    alpha = np.linspace(0.0, 1.0, count, dtype=np.float64)
    blended = (
        hidden[:count].astype(np.float64) * (1.0 - alpha)
        + true[:count].astype(np.float64) * alpha
    )
    out[:count] = _to_pcm16(blended)
    return out
