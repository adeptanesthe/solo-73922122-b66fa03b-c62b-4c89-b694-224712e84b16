"""Unit tests for the NumPy concealment primitives."""

from __future__ import annotations

import numpy as np
import pytest

from app.concealment import (
    concealment_for,
    crossfade_to_true,
    estimate_period,
)
from app.constants import (
    FADE_SAMPLES,
    FRAME_SAMPLES,
    HISTORY_SAMPLES,
    MAX_PERIOD_SAMPLES,
    MIN_PERIOD_SAMPLES,
)
from helpers import tone


def test_finds_fundamental_period():
    history = tone(160, HISTORY_SAMPLES)
    estimate = estimate_period(history)
    assert estimate.reason == "periodic"
    assert estimate.period_samples == 160
    assert estimate.peak is not None and estimate.peak > 0.99


def test_period_uses_only_recent_40ms():
    # Old loud period 100 followed by a fresh period-50 segment: only the
    # last 640 samples may inform the estimate.
    old = tone(100, 2_000, amplitude=2_000.0)
    fresh = tone(50, HISTORY_SAMPLES, amplitude=12_000.0)
    history = np.concatenate([old, fresh])
    estimate = estimate_period(history)
    assert estimate.reason == "periodic"
    assert estimate.period_samples == 50


def test_tie_prefers_shortest_lag():
    # A period-100 tone also correlates perfectly at lag 200.
    estimate = estimate_period(tone(100, HISTORY_SAMPLES))
    assert estimate.reason == "periodic"
    assert estimate.period_samples == 100


def test_insufficient_history_is_silence():
    history = tone(160, HISTORY_SAMPLES - 1)
    estimate = estimate_period(history)
    assert estimate.reason == "insufficient_history"
    result = concealment_for(history)
    assert result.applied is False
    assert np.all(result.samples == 0)


def test_flat_signal_has_undefined_autocorrelation():
    estimate = estimate_period(np.full(HISTORY_SAMPLES, 1234, dtype=np.int16))
    assert estimate.reason == "autocorr_undefined"
    result = concealment_for(np.full(HISTORY_SAMPLES, 1234, dtype=np.int16))
    assert np.all(result.samples == 0)
    assert result.applied is False


def test_zero_history_is_silence():
    result = concealment_for(np.zeros(HISTORY_SAMPLES, dtype=np.int16))
    assert np.all(result.samples == 0)


def test_noise_has_low_correlation_and_is_silenced():
    rng = np.random.default_rng(1234)
    noise = rng.integers(-8000, 8000, size=HISTORY_SAMPLES, dtype=np.int16)
    estimate = estimate_period(noise)
    assert estimate.peak is not None
    assert estimate.peak < 0.6
    assert estimate.reason == "low_correlation"
    result = concealment_for(noise)
    assert result.applied is False
    assert np.all(result.samples == 0)


def test_concealment_repeats_last_period_at_half_amplitude():
    period = 160
    history = tone(period, HISTORY_SAMPLES, amplitude=10_000.0)
    result = concealment_for(history)
    assert result.applied is True
    assert result.period_samples == period

    expected_period = history[-period:].astype(np.float64) * 0.5
    expected = np.tile(expected_period, 2)[:FRAME_SAMPLES]
    np.testing.assert_allclose(result.samples, np.rint(expected), atol=1)


def test_concealment_tiles_and_truncates_to_one_frame():
    period = 70  # 320 / 70 is not an integer.
    history = tone(period, HISTORY_SAMPLES, amplitude=9_000.0)
    result = concealment_for(history)
    assert result.applied is True
    assert result.samples.size == FRAME_SAMPLES
    expected_period = history[-period:].astype(np.float64) * 0.5
    expected = np.tile(expected_period, 5)[:FRAME_SAMPLES]
    np.testing.assert_allclose(result.samples, np.rint(expected), atol=1)


def test_search_bounds_are_inclusive():
    low = tone(MIN_PERIOD_SAMPLES, HISTORY_SAMPLES)
    high = tone(MAX_PERIOD_SAMPLES, HISTORY_SAMPLES)
    assert estimate_period(low).period_samples == MIN_PERIOD_SAMPLES
    assert estimate_period(high).period_samples == MAX_PERIOD_SAMPLES


def test_autocorrelation_stays_inside_declared_window():
    # A period below 40 samples must not be reported; no lag in range is a
    # clean match for a 32-sample tone, but the result must still obey bounds.
    estimate = estimate_period(tone(32, HISTORY_SAMPLES))
    if estimate.period_samples is not None:
        assert MIN_PERIOD_SAMPLES <= estimate.period_samples <= MAX_PERIOD_SAMPLES


def test_crossfade_moves_linearly_from_concealment_to_truth():
    hidden = np.full(FRAME_SAMPLES, 1000, dtype=np.int16)
    true = np.full(FRAME_SAMPLES, -1000, dtype=np.int16)
    out = crossfade_to_true(hidden, true)

    assert out.dtype == np.int16
    assert out.size == FRAME_SAMPLES
    # Sample 0 is pure concealment; the fade completes by the 5 ms boundary.
    assert out[0] == 1000
    assert out[FADE_SAMPLES - 1] == -1000
    # Everything after the fade is byte-for-byte the real frame.
    np.testing.assert_array_equal(out[FADE_SAMPLES:], true[FADE_SAMPLES:])
    # Linear in between, within rounding.
    alpha = np.linspace(0.0, 1.0, FADE_SAMPLES)
    expected = 1000 * (1 - alpha) + (-1000) * alpha
    np.testing.assert_allclose(out[:FADE_SAMPLES], np.rint(expected), atol=1)


def test_crossfade_rejects_mismatched_shapes():
    with pytest.raises(ValueError):
        crossfade_to_true(
            np.zeros(FRAME_SAMPLES, dtype=np.int16),
            np.zeros(FRAME_SAMPLES - 1, dtype=np.int16),
        )
