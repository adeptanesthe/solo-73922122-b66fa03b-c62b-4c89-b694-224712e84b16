"""NumPy concealment primitives with an incremental period estimator.

All arrays are mono int16 samples.  The history handed to these functions is
the *played* signal (real frames plus earlier concealment), exactly as it was
emitted, which makes concealment reproducible and immune to late packets.

Incremental autocorrelation
---------------------------
A period estimate scans the last 40 ms (640 samples) of played audio.  During
consecutive loss the estimates of two successive pulls therefore use windows
that are offset by exactly one 20 ms frame (320 samples): they share the 320
samples in the middle.  The raw (mean *not* yet removed) sample-product sums
of that shared region are identical for every lag, so they are computed once
and reused.

A window ``W = [H | T]`` of two 320-sample halves has, for lag L, the raw
product sum

    num(W, L) = D(H, L) + C(H, T, L) + D(T, L)

where ``D(X, L) = sum X[i] X[i+L]`` (i = 0 .. 319-L) is the inside-half
product sum and ``C(H, T, L) = sum H[320-L+k] T[k]`` (k = 0 .. L-1) is the
L-product cross term over the half boundary.  When the window slides to
``[T | F]`` the D(T) terms carry over untouched, so

    num([T|F], L) = num([H|T], L)
                    - D(H, L) - C(H, T, L)      # leaving contributions
                    + C(T, F, L) + D(F, L)      # entering contributions

Only the newly played frame F (and the retained tail T) are touched; the
overlap-frame raw products are never recomputed.  All raw sums accumulate as
exact int64 integers.  The mean removal and the per-lag normalisation are
then recomputed from the *new* window alone, using its own mean and the
energy of the overlapping samples on each side, so the normalised
correlation is the same one a from-scratch search produces.

The retained state has a constant size (one value per searched lag plus one
frame of samples); it never grows with the length of the stream.
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

# A full 40 ms window is two 20 ms frames.
_HALF = HISTORY_SAMPLES // 2  # 320
_LAGS = np.arange(MIN_PERIOD_SAMPLES, MAX_PERIOD_SAMPLES + 1)
_LAG_COUNT = int(_LAGS.size)


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


def _self_products(frame: np.ndarray) -> np.ndarray:
    """D(X, L): raw inside-half product sums, one per searched lag.

    Accumulated as exact int64 integers (an int16 product is at most
    ~1.1e9 and at most 280 of them are added per lag).  All lags are
    evaluated in one vectorised ``correlate`` call.
    """

    frame_i64 = frame.astype(np.int64)
    correlation = np.correlate(frame_i64, frame_i64, mode="full")
    return correlation[_HALF - 1 + _LAGS]


def _cross_products(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """C(A, B, L): the L raw products straddling the half boundary.

    C(A, B, L) = sum_{k=0..L-1} A[320-L+k] * B[k]

    Writing i = 320-L+k turns this into the cross-correlation of A and B at
    displacement 320-L, which the full-length integer correlation table
    contains directly.  All lags are therefore read from one vectorised
    ``correlate`` call; the int64 accumulation is exact because every partial
    product is an int16 product and at most 200 of them are summed.
    """

    first_i64 = first.astype(np.int64)
    second_i64 = second.astype(np.int64)
    correlation = np.correlate(first_i64, second_i64, mode="full")
    # Full table index for displacement d = 320-L is (320-1) + d = 639-L.
    return correlation[2 * _HALF - 1 - _LAGS]


def _period_from_raw(
    window_int: np.ndarray, raw_numerator: np.ndarray
) -> PeriodEstimate:
    """Mean-remove, normalise per lag and apply the selection rules.

    Given the raw (non-mean-removed) product sums of the 640-sample window,
    everything else is derived from the *new* window alone.  With global mean
    m and, for lag L, the overlapping left/right subsets::

        sum (x[i]-m)(x[i+L]-m)
            = raw(L) - m*(S_left(L) + S_right(L)) + (N-L)*m^2
        var_left(L)  = E_left(L)  - 2*m*S_left(L)  + (N-L)*m^2
        var_right(L) = E_right(L) - 2*m*S_right(L) + (N-L)*m^2

    A lag without energy on either side is undefined and skipped.  The 0.6
    gate and the 1e-9 shortest-lag tie rule are unchanged.
    """

    window = window_int.astype(np.float64)
    size = window.size
    mean = float(window.mean())

    cumulative = np.concatenate(([0.0], np.cumsum(window)))
    cumulative_sq = np.concatenate(([0.0], np.cumsum(window * window)))
    total_sum = float(cumulative[-1])
    total_energy = float(cumulative_sq[-1])

    pair_counts = size - _LAGS
    sum_left = cumulative[pair_counts]
    sum_right = total_sum - cumulative[_LAGS]
    energy_left_raw = cumulative_sq[pair_counts]
    energy_right_raw = total_energy - cumulative_sq[_LAGS]

    mean_sq = mean * mean
    counts_f = pair_counts.astype(np.float64)
    numerator = (
        raw_numerator.astype(np.float64)
        - mean * (sum_left + sum_right)
        + counts_f * mean_sq
    )
    energy_left = np.maximum(
        energy_left_raw - 2.0 * mean * sum_left + counts_f * mean_sq, 0.0
    )
    energy_right = np.maximum(
        energy_right_raw - 2.0 * mean * sum_right + counts_f * mean_sq, 0.0
    )

    defined = (energy_left > 0.0) & (energy_right > 0.0)
    if not np.any(defined):
        return PeriodEstimate(None, None, "autocorr_undefined")

    peaks = np.full(_LAG_COUNT, -np.inf, dtype=np.float64)
    peaks[defined] = numerator[defined] / np.sqrt(
        energy_left[defined] * energy_right[defined]
    )
    # Rounding makes exact ties land within ~1e-15 of each other; treat that
    # band as a tie and keep the shortest lag (the fundamental).
    best_peak = float(np.max(peaks))
    tied = np.flatnonzero(peaks >= best_peak - 1e-9)
    best_lag = int(_LAGS[int(tied[0])])
    best_peak = min(best_peak, 1.0)

    if best_peak < MIN_PEAK_CORRELATION:
        return PeriodEstimate(None, best_peak, "low_correlation")
    return PeriodEstimate(best_lag, best_peak, "periodic")


def estimate_period(history: np.ndarray) -> PeriodEstimate:
    """Find the strongest normalised autocorrelation lag in the last 40 ms.

    Stateless from-scratch search; it uses the same raw-product primitives as
    the incremental estimator so both paths share their integer arithmetic.
    The autocorrelation is mean-removed and normalised per lag, using the
    overlapping samples only.
    """

    x = np.asarray(history, dtype=np.int16)
    if x.ndim != 1:
        raise ValueError("history must be one-dimensional")
    if x.size < HISTORY_SAMPLES:
        return PeriodEstimate(None, None, "insufficient_history")

    window = x[-HISTORY_SAMPLES:]
    head, tail = window[:_HALF], window[_HALF:]
    raw_numerator = (
        _self_products(head)
        + _cross_products(head, tail)
        + _self_products(tail)
    )
    return _period_from_raw(window, raw_numerator)


class IncrementalPeriodEstimator:
    """Reuses raw autocorrelation products between one-frame-shifted windows.

    The estimator only does work for concealment / recovery pulls; plain true
    frames that are played straight through never trigger a correlation
    precomputation.  State is keyed by the absolute played-sample position of
    the window, so a window that merely *looks* like the previous one after a
    longer gap can never be mistaken for an adjacent (one-frame) shift.

    Retained state (all bounded, independent of stream length):

    * ``window_end``   - absolute sample index just past the cached window;
    * ``tail``         - cached window's last 320 played int16 samples;
    * ``raw_products`` - one int64 raw product sum per searched lag;
    * ``leaving``      - per-lag products removed on the next one-frame shift.
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Drop every intermediate result."""

        self._window_end: int | None = None
        self._tail: np.ndarray | None = None
        self._raw_products: np.ndarray | None = None
        self._leaving: np.ndarray | None = None

    @property
    def window_end(self) -> int | None:
        """Absolute end position of the cached window, if any."""

        return self._window_end

    def estimate(
        self, history: np.ndarray, played_samples: int
    ) -> PeriodEstimate:
        """Estimate the period of the latest 40 ms of played audio.

        ``played_samples`` is the total number of samples emitted so far
        (including the frame about to be analysed); the window's absolute end
        position is exactly ``played_samples``.
        """

        x = np.asarray(history, dtype=np.int16)
        if x.ndim != 1:
            raise ValueError("history must be one-dimensional")
        if x.size < HISTORY_SAMPLES:
            # Nothing to cache yet; keep any earlier state untouched.
            return PeriodEstimate(None, None, "insufficient_history")

        window = x[-HISTORY_SAMPLES:]
        raw_products = self._raw_products_for(window, played_samples)
        return _period_from_raw(window, raw_products)

    def _raw_products_for(
        self, window: np.ndarray, window_end: int
    ) -> np.ndarray:
        head = window[:_HALF]
        tail = window[_HALF:]
        adjacent = (
            self._window_end is not None
            and window_end - self._window_end == _HALF
            and self._tail is not None
            and self._raw_products is not None
            and self._leaving is not None
            and np.array_equal(head, self._tail)
        )
        if adjacent:
            # Windows overlap by one frame: the shared frame's inside
            # products carry over; only the leaving and entering terms are
            # computed, each touching at most one frame.
            cross_in = _cross_products(self._tail, tail)
            raw_products = (
                self._raw_products
                - self._leaving
                + cross_in
                + _self_products(tail)
            )
            leaving = _self_products(self._tail) + cross_in
        else:
            # First estimate, or a gap of two or more frames since the last
            # estimate (e.g. a run of plain true frames): rebuild from the
            # current window and start a fresh incremental chain.
            self_head = _self_products(head)
            cross = _cross_products(head, tail)
            raw_products = self_head + cross + _self_products(tail)
            leaving = self_head + cross

        # The next adjacent window, if any, must start at this tail.
        self._window_end = int(window_end)
        self._tail = np.array(tail, dtype=np.int16, copy=True)
        self._raw_products = np.asarray(raw_products, dtype=np.int64).copy()
        self._leaving = np.asarray(leaving, dtype=np.int64).copy()
        return self._raw_products


def concealment_from_estimate(
    history: np.ndarray, estimate: PeriodEstimate
) -> Concealment:
    """Build one 20 ms concealment frame from an existing period estimate."""

    x = np.asarray(history, dtype=np.int16)
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


def concealment_for(history: np.ndarray) -> Concealment:
    """Build one 20 ms concealment frame from the last 40 ms of played audio.

    * Too little history, undefined autocorrelation, or a peak below 0.6
      yields silence.
    * Otherwise the last detected period is tiled over 320 samples at half
      amplitude.  Because the emitted frame becomes part of the history, a
      run of missing frames recursively repeats (and halves) the newest
      period.
    """

    return concealment_from_estimate(history, estimate_period(history))


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
