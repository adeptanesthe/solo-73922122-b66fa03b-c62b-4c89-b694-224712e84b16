"""Tests for the incremental autocorrelation estimator.

These cover the optimisation itself: raw-product reuse across windows that
shift by exactly one frame, cold-path fallback for larger jumps, the fact
that plain true frames never prime the correlation cache, reset semantics,
bounded retained state, and end-to-end equivalence with the stateless
from-scratch search under random loss patterns.
"""

from __future__ import annotations

import numpy as np

from app.concealment import (
    IncrementalPeriodEstimator,
    _cross_products,
    _self_products,
    concealment_for,
    estimate_period,
)
from app.constants import (
    FRAME_SAMPLES,
    HISTORY_SAMPLES,
    MAX_PERIOD_SAMPLES,
    MIN_PERIOD_SAMPLES,
)
from app.engine import PlayoutEngine
from helpers import tone, tone_frames


def test_raw_product_primitives_are_exact_int64():
    # Full-scale int16 inputs would defeat float32/rounded-FFT accumulation;
    # the vectorised primitives must match direct integer dot products.
    rng = np.random.default_rng(11)
    for _ in range(50):
        a = rng.integers(-32768, 32767, size=FRAME_SAMPLES, dtype=np.int16)
        b = rng.integers(-32768, 32767, size=FRAME_SAMPLES, dtype=np.int16)
        d = _self_products(a)
        c = _cross_products(a, b)
        assert d.dtype == np.int64 and c.dtype == np.int64
        for index, lag in enumerate(
            range(MIN_PERIOD_SAMPLES, MAX_PERIOD_SAMPLES + 1)
        ):
            assert d[index] == np.dot(
                a[: FRAME_SAMPLES - lag].astype(np.int64),
                a[lag:].astype(np.int64),
            )
            assert c[index] == np.dot(
                a[FRAME_SAMPLES - lag :].astype(np.int64),
                b[:lag].astype(np.int64),
            )


def _signal(seed: int, samples: int) -> np.ndarray:
    """Deterministic int16 material mixing tones, drift and a noise burst."""

    rng = np.random.default_rng(seed)
    t = np.arange(samples)
    phase = np.cumsum(
        2 * np.pi / (140 + 25 * np.sin(2 * np.pi * t / 4000))
    )
    voice = 9000 * np.sin(phase) + 3000 * np.sin(2 * phase + 0.7)
    burst = rng.integers(-2500, 2500, size=samples)
    return np.clip(np.rint(voice + burst), -32768, 32767).astype(np.int16)


def test_rolling_estimates_match_from_scratch_search():
    signal = _signal(1, HISTORY_SAMPLES + FRAME_SAMPLES * 24)
    estimator = IncrementalPeriodEstimator()

    for k in range(25):
        end = HISTORY_SAMPLES + k * FRAME_SAMPLES
        history = signal[:end]
        got = estimator.estimate(history, played_samples=end)
        expected = estimate_period(history[-HISTORY_SAMPLES:])

        assert got.reason == expected.reason
        assert got.period_samples == expected.period_samples
        if expected.peak is not None:
            assert got.peak is not None
            assert abs(got.peak - expected.peak) < 1e-12


def test_rolling_estimates_match_on_pure_tones_and_silence():
    for period in (40, 70, 100, 160, 200):
        signal = tone(period, HISTORY_SAMPLES + FRAME_SAMPLES * 8)
        estimator = IncrementalPeriodEstimator()
        for k in range(9):
            end = HISTORY_SAMPLES + k * FRAME_SAMPLES
            got = estimator.estimate(signal[:end], end)
            expected = estimate_period(signal[end - HISTORY_SAMPLES : end])
            assert got.period_samples == expected.period_samples == period

    # All-zero stream stays undefined/silence through every shift; the
    # incremental chain still advances (silent frames are played history).
    zeros = np.zeros(HISTORY_SAMPLES + FRAME_SAMPLES * 4, dtype=np.int16)
    estimator = IncrementalPeriodEstimator()
    for k in range(5):
        end = HISTORY_SAMPLES + k * FRAME_SAMPLES
        got = estimator.estimate(zeros[:end], end)
        assert got.reason in {"autocorr_undefined", "insufficient_history"}
    assert estimator.window_end == HISTORY_SAMPLES + 4 * FRAME_SAMPLES


def test_non_adjacent_window_falls_back_to_cold_path():
    signal = _signal(2, HISTORY_SAMPLES + FRAME_SAMPLES * 6)
    estimator = IncrementalPeriodEstimator()
    estimator.estimate(signal[:HISTORY_SAMPLES], HISTORY_SAMPLES)

    # Two-frame jump: the cached window cannot be reused.
    end = HISTORY_SAMPLES + 2 * FRAME_SAMPLES
    got = estimator.estimate(signal[:end], end)
    expected = estimate_period(signal[end - HISTORY_SAMPLES : end])
    assert got.period_samples == expected.period_samples
    assert got.reason == expected.reason
    # The chain restarts from the new window.
    assert estimator.window_end == end


def test_lookalike_window_after_gap_is_not_treated_as_adjacent():
    # A periodic tone makes far-apart windows numerically identical; the
    # positional key must still reject the stale cache.
    signal = tone(160, HISTORY_SAMPLES + FRAME_SAMPLES * 5)
    estimator = IncrementalPeriodEstimator()
    estimator.estimate(signal[:HISTORY_SAMPLES], HISTORY_SAMPLES)

    end = HISTORY_SAMPLES + 3 * FRAME_SAMPLES
    window = signal[end - HISTORY_SAMPLES : end]
    assert np.array_equal(window[:FRAME_SAMPLES], estimator._tail)
    got = estimator.estimate(window, end)
    expected = estimate_period(window)
    assert got.period_samples == expected.period_samples
    assert estimator.window_end == end


def test_true_frames_skip_correlation_precomputation():
    engine = PlayoutEngine()
    frames = tone_frames(10)
    for seq, frame in enumerate(frames):
        engine.ingest(seq, frame)

    # Only true frames pass through: the estimator stays untouched.
    for _ in range(10):
        result = engine.playout()
        assert result.kind == "true"
    assert engine._estimator.window_end is None
    assert engine._estimator._raw_products is None
    assert engine._estimator._leaving is None
    assert engine._estimator._tail is None


def test_estimator_engages_on_concealment_and_recovery():
    engine = PlayoutEngine()
    frames = tone_frames(6)
    engine.ingest(0, frames[0])
    engine.ingest(1, frames[1])
    engine.ingest(4, frames[4])
    engine.ingest(5, frames[5])

    engine.playout(), engine.playout()
    concealed = engine.playout()  # seq 2 lost
    assert concealed.kind == "concealed"
    cached_end = engine._estimator.window_end
    assert cached_end == 3 * FRAME_SAMPLES

    recovered = engine.playout()  # seq 3 lost too -> another concealment
    assert recovered.kind == "concealed"
    # One-frame shift: the cache advanced by exactly one frame.
    assert engine._estimator.window_end == cached_end + FRAME_SAMPLES

    real = engine.playout()  # seq 4 present, had_gap -> recovered
    assert real.kind == "recovered"
    assert engine._estimator.window_end == cached_end + 2 * FRAME_SAMPLES


def test_reset_clears_estimator_intermediate_results():
    engine = PlayoutEngine()
    frames = tone_frames(4)
    engine.ingest(0, frames[0])
    engine.ingest(1, frames[1])
    engine.playout(), engine.playout()
    engine.playout()  # concealment primes the cache

    assert engine._estimator.window_end is not None
    engine.reset()
    assert engine._estimator.window_end is None
    assert engine._estimator._tail is None
    assert engine._estimator._raw_products is None
    assert engine._estimator._leaving is None
    assert engine.snapshot()["played_samples"] == 0


def test_retained_state_does_not_grow_with_stream_length():
    engine = PlayoutEngine()
    rng = np.random.default_rng(7)
    # Long alternating run of true frames and single-frame losses.
    total = 400
    frames = [
        rng.integers(-8000, 8000, FRAME_SAMPLES, dtype=np.int16)
        for _ in range(total)
    ]
    for seq, frame in enumerate(frames):
        if seq % 7 != 3:  # periodic single-frame losses
            engine.ingest(seq, frame)

    sizes = []
    for _ in range(total):
        engine.playout()
        sizes.append(
            (
                0
                if engine._estimator._raw_products is None
                else engine._estimator._raw_products.size,
                0
                if engine._estimator._tail is None
                else engine._estimator._tail.size,
                engine.snapshot()["history_samples"],
            )
        )

    product_size, tail_size, history_size = sizes[-1]
    assert product_size == 200 - 40 + 1  # one entry per searched lag
    assert tail_size == FRAME_SAMPLES
    assert history_size <= HISTORY_SAMPLES
    # Every later snapshot has the same footprint as the first primed one.
    primed = [size for size in sizes if size[0] > 0]
    assert set(primed) == {(product_size, tail_size, HISTORY_SAMPLES)}


def test_end_to_end_matches_stateless_concealment_under_random_losses():
    rng = np.random.default_rng(99)
    signal = _signal(99, FRAME_SAMPLES * 60)
    frames = [
        signal[i * FRAME_SAMPLES : (i + 1) * FRAME_SAMPLES].copy()
        for i in range(60)
    ]

    engine = PlayoutEngine()
    reference = PlayoutEngine()
    # The reference engine is forced onto the stateless from-scratch search.
    reference._estimator = None  # type: ignore[assignment]

    def reference_concealment(history, played_samples):  # noqa: ANN001
        return concealment_for(history)

    reference._concealment_for = reference_concealment  # type: ignore[method-assign]

    lost = set(rng.choice(60, size=18, replace=False).tolist())
    for seq, frame in enumerate(frames):
        if seq not in lost:
            engine.ingest(seq, frame)
            reference.ingest(seq, frame)

    for _ in range(60):
        got = engine.playout()
        expected = reference.playout()
        assert got.kind == expected.kind
        assert got.period_samples == expected.period_samples
        assert got.conceal_reason == expected.conceal_reason
        np.testing.assert_array_equal(got.samples, expected.samples)


def test_ingestion_keeps_working_between_incremental_estimates():
    # The estimator runs inside the play lock only; /frames must stay
    # available while consecutive concealments are being computed.
    engine = PlayoutEngine()
    frames = tone_frames(6)
    engine.ingest(0, frames[0])
    engine.ingest(1, frames[1])
    engine.playout(), engine.playout()

    # Two consecutive losses: both estimates share one-frame-shifted windows.
    first = engine.playout()
    second = engine.playout()
    assert first.kind == "concealed" and second.kind == "concealed"
    # Packets for later sequences are accepted all the while.
    assert engine.ingest(5, frames[5]).status == "accepted"
    assert engine.snapshot()["pending_seqs"] == [5]
