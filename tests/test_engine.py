"""Behavioural tests for the ordered playout engine."""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from app.concealment import concealment_for, crossfade_to_true
from app.constants import FADE_SAMPLES, FRAME_SAMPLES, HISTORY_SAMPLES
from app.engine import PlayoutEngine
from helpers import tone, tone_frames


def frames(count: int, period: int = 160) -> list[np.ndarray]:
    return tone_frames(count, period)


def feed(engine: PlayoutEngine, seq: int, frame: np.ndarray) -> str:
    return engine.ingest(seq, frame).status


def test_plays_from_zero_in_order():
    engine = PlayoutEngine()
    f = frames(3)
    for i, frame in enumerate(f):
        assert feed(engine, i, frame) == "accepted"

    results = [engine.playout() for _ in range(3)]
    assert [r.seq for r in results] == [0, 1, 2]
    assert [r.kind for r in results] == ["true"] * 3
    for result, frame in zip(results, f):
        np.testing.assert_array_equal(result.samples, frame)


def test_out_of_order_arrival_is_ordered_on_playout():
    engine = PlayoutEngine()
    f = frames(4)
    for seq in (3, 1, 0, 2):
        assert feed(engine, seq, f[seq]) == "accepted"

    results = [engine.playout() for _ in range(4)]
    assert [r.seq for r in results] == [0, 1, 2, 3]
    for result, frame in zip(results, f):
        np.testing.assert_array_equal(result.samples, frame)
    assert engine.snapshot()["stats"]["concealed_frames"] == 0


def test_missing_frame_is_concealed_from_played_history():
    engine = PlayoutEngine()
    f = frames(4)
    feed(engine, 0, f[0])
    feed(engine, 1, f[1])
    feed(engine, 3, f[3])  # seq 2 is lost

    r0, r1, r2 = (engine.playout() for _ in range(3))
    assert (r0.kind, r1.kind) == ("true", "true")
    assert r2.kind == "concealed"
    assert r2.period_samples == 160
    assert r2.peak is not None and r2.peak > 0.99

    history_after_1 = np.concatenate(f[:2])[-HISTORY_SAMPLES:]
    expected = concealment_for(history_after_1)
    np.testing.assert_array_equal(r2.samples, expected.samples)


def test_consecutive_missing_frames_each_repeat_half_amplitude():
    engine = PlayoutEngine()
    f = frames(2)
    feed(engine, 0, f[0])
    feed(engine, 1, f[1])

    r0, r1 = engine.playout(), engine.playout()
    r2, r3 = engine.playout(), engine.playout()  # 2 and 3 are both lost
    assert (r2.kind, r3.kind) == ("concealed", "concealed")

    history = np.concatenate(f[:2])[-HISTORY_SAMPLES:]
    c2 = concealment_for(history)
    np.testing.assert_array_equal(r2.samples, c2.samples)

    history_3 = np.concatenate([history, c2.samples])[-HISTORY_SAMPLES:]
    c3 = concealment_for(history_3)
    np.testing.assert_array_equal(r3.samples, c3.samples)

    # Successive concealment repeats a period already halved, so peak level
    # keeps dropping.
    assert np.max(np.abs(r3.samples.astype(np.int32))) < np.max(
        np.abs(r2.samples.astype(np.int32))
    )


def test_first_recovered_frame_fades_only_the_first_5ms():
    engine = PlayoutEngine()
    f = frames(4)
    feed(engine, 0, f[0])
    feed(engine, 1, f[1])
    # seq 2 is missing, seq 3 is present.
    feed(engine, 3, f[3])

    engine.playout(), engine.playout()
    missing = engine.playout()
    assert missing.kind == "concealed"

    before_recovery = engine.snapshot()["history"]
    hypothetical = concealment_for(before_recovery)
    recovered = engine.playout()

    assert recovered.seq == 3
    assert recovered.kind == "recovered"
    assert recovered.crossfade_samples == FADE_SAMPLES
    np.testing.assert_array_equal(
        recovered.samples,
        crossfade_to_true(hypothetical.samples, f[3]),
    )
    # Fade region is a blend; the rest is the true frame untouched.
    assert recovered.samples[0] == hypothetical.samples[0]
    np.testing.assert_array_equal(recovered.samples[FADE_SAMPLES:], f[3][FADE_SAMPLES:])


def test_later_frames_play_normally_after_recovery():
    engine = PlayoutEngine()
    f = frames(5)
    feed(engine, 0, f[0])
    feed(engine, 1, f[1])
    feed(engine, 3, f[3])
    feed(engine, 4, f[4])

    kinds = [engine.playout().kind for _ in range(5)]
    assert kinds == ["true", "true", "concealed", "recovered", "true"]


def test_insufficient_history_or_silence_yields_silence():
    engine = PlayoutEngine()
    # Nothing received: no history at all.
    r0 = engine.playout()
    assert (r0.kind, r0.conceal_reason) == ("silence", "insufficient_history")
    assert np.all(r0.samples == 0)

    # The next real frame follows a gap whose hypothetical concealment was
    # silence, so it fades in from silence rather than from a period.
    feed(engine, 1, tone(160, FRAME_SAMPLES))
    r1 = engine.playout()
    assert (r1.kind, r1.conceal_reason) == ("recovered", "insufficient_history")
    assert r1.samples[0] == 0
    np.testing.assert_array_equal(
        r1.samples[FADE_SAMPLES:], tone(160, FRAME_SAMPLES)[FADE_SAMPLES:]
    )


def test_low_correlation_and_undefined_autocorrelation_yield_silence():
    rng = np.random.default_rng(7)
    engine = PlayoutEngine()
    feed(engine, 0, rng.integers(-6000, 6000, FRAME_SAMPLES, dtype=np.int16))
    feed(engine, 1, rng.integers(-6000, 6000, FRAME_SAMPLES, dtype=np.int16))
    engine.playout(), engine.playout()
    noise_missing = engine.playout()
    assert noise_missing.kind == "silence"
    assert noise_missing.conceal_reason == "low_correlation"

    flat = PlayoutEngine()
    feed(flat, 0, np.full(FRAME_SAMPLES, 500, dtype=np.int16))
    feed(flat, 1, np.full(FRAME_SAMPLES, 500, dtype=np.int16))
    flat.playout(), flat.playout()
    flat_missing = flat.playout()
    assert flat_missing.kind == "silence"
    assert flat_missing.conceal_reason == "autocorr_undefined"


def test_duplicate_sequence_keeps_first_received_copy():
    engine = PlayoutEngine()
    first = tone(160, FRAME_SAMPLES, amplitude=8_000.0)
    second = tone(120, FRAME_SAMPLES, amplitude=15_000.0)
    assert feed(engine, 0, first) == "accepted"
    assert feed(engine, 0, second) == "duplicate"
    assert feed(engine, 0, first) == "duplicate"

    out = engine.playout()
    np.testing.assert_array_equal(out.samples, first)
    assert engine.snapshot()["stats"]["duplicate_frames"] == 2


def test_late_packet_cannot_change_played_audio_or_later_concealment():
    control = PlayoutEngine()
    candidate = PlayoutEngine()
    f = frames(2)
    for engine in (control, candidate):
        feed(engine, 0, f[0])
        feed(engine, 1, f[1])
        assert engine.playout().kind == "true"
        assert engine.playout().kind == "true"
        assert engine.playout().kind == "concealed"  # seq 2 missing

    played_before = candidate.snapshot()["history"].copy()
    loud_late = np.full(FRAME_SAMPLES, 30_000, dtype=np.int16)
    result = candidate.ingest(2, loud_late)
    assert result.status == "late_ignored"
    np.testing.assert_array_equal(candidate.snapshot()["history"], played_before)

    # seq 3 missing on both; the concealment must be identical.
    r3_control = control.playout()
    r3_candidate = candidate.playout()
    assert r3_candidate.seq == 3
    np.testing.assert_array_equal(r3_candidate.samples, r3_control.samples)
    assert candidate.snapshot()["stats"]["late_frames"] == 1


def test_late_packet_after_gap_does_not_alter_recovery():
    control = PlayoutEngine()
    candidate = PlayoutEngine()
    f = frames(4)
    for engine in (control, candidate):
        feed(engine, 0, f[0])
        feed(engine, 1, f[1])
        engine.playout(), engine.playout()
        engine.playout()  # conceal seq 2

    late = candidate.ingest(2, tone(80, FRAME_SAMPLES, amplitude=20_000.0))
    assert late.status == "late_ignored"
    feed(candidate, 3, f[3])

    # Control has no frame 3, so its seq 3 is the hypothetical concealment
    # that the candidate must blend from.
    hypothetical = control.playout()
    assert hypothetical.seq == 3
    assert hypothetical.kind == "concealed"

    recovered = candidate.playout()
    assert recovered.seq == 3
    assert recovered.kind == "recovered"
    assert recovered.samples[0] == hypothetical.samples[0]
    np.testing.assert_array_equal(
        recovered.samples[FADE_SAMPLES:], f[3][FADE_SAMPLES:]
    )


def test_ingest_continues_while_concealment_is_running():
    class SlowConcealment(PlayoutEngine):
        started = threading.Event()
        release = threading.Event()

        def _concealment_for(self, history):
            self.started.set()
            self.release.wait(timeout=5.0)
            return super()._concealment_for(history)

    engine = SlowConcealment()
    f = frames(4)
    feed(engine, 0, f[0])
    feed(engine, 1, f[1])  # seq 2 is missing, so the third pull conceals

    def pull():
        return engine.playout()

    with ThreadPoolExecutor(max_workers=1) as pool:
        engine.playout()
        engine.playout()
        pull_future = pool.submit(pull)
        assert engine.started.wait(timeout=2.0), "concealment did not start"

        start = time.monotonic()
        status = feed(engine, 3, f[3])
        status_3 = feed(engine, 3, f[3])  # second copy loses
        elapsed = time.monotonic() - start

        engine.release.set()
        concealed = pull_future.result(timeout=5.0)

    # Ingestion did not wait for the 5 s concealment gate.
    assert elapsed < 0.5
    assert status == "accepted"
    assert status_3 == "duplicate"
    assert concealed.kind == "concealed"


def test_concurrent_pullouts_each_return_one_ordered_frame():
    engine = PlayoutEngine()
    f = frames(64)
    for seq, frame in enumerate(f):
        feed(engine, seq, frame)

    barrier = threading.Barrier(16)

    def pull(_: int):
        barrier.wait(timeout=5.0)
        return engine.playout()

    with ThreadPoolExecutor(max_workers=16) as pool:
        results = list(pool.map(pull, range(16)))

    seqs = sorted(r.seq for r in results)
    assert seqs == list(range(16)), "every pull must return a distinct frame"
    assert all(r.kind == "true" for r in results)
    for result in results:
        np.testing.assert_array_equal(result.samples, f[result.seq])
    assert engine.snapshot()["next_seq"] == 16


def test_concurrent_pulls_across_a_gap_return_sequential_frames():
    engine = PlayoutEngine()
    f = frames(24)
    for seq, frame in enumerate(f):
        if seq in (5, 6):
            continue  # lost
        feed(engine, seq, frame)

    def pull(_: int):
        return engine.playout()

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(pull, range(12)))

    by_seq = {r.seq: r for r in results}
    assert sorted(by_seq) == list(range(12))
    assert by_seq[5].kind == "concealed"
    assert by_seq[6].kind == "concealed"
    assert by_seq[7].kind == "recovered"
    for seq in (0, 1, 2, 3, 4, 8, 9, 10, 11):
        np.testing.assert_array_equal(by_seq[seq].samples, f[seq])


def test_reset_clears_everything():
    engine = PlayoutEngine()
    feed(engine, 2, tone(160, FRAME_SAMPLES))
    engine.playout()
    engine.reset()

    state = engine.snapshot()
    assert state["next_seq"] == 0
    assert state["history_samples"] == 0
    assert state["pending_frames"] == 0
    assert all(value == 0 for value in state["stats"].values())


def test_invalid_frames_are_rejected():
    engine = PlayoutEngine()
    with pytest.raises(ValueError):
        engine.ingest(0, np.zeros(FRAME_SAMPLES - 1, dtype=np.int16))
    with pytest.raises(ValueError):
        engine.ingest(0, np.zeros(FRAME_SAMPLES, dtype=np.float32))
    with pytest.raises(ValueError):
        engine.ingest(-1, np.zeros(FRAME_SAMPLES, dtype=np.int16))


def test_reordering_buffer_is_bounded():
    from app.constants import MAX_PENDING_FRAMES
    from app.engine import BacklogFullError

    engine = PlayoutEngine()
    frame = tone(160, FRAME_SAMPLES)
    for seq in range(MAX_PENDING_FRAMES):
        assert feed(engine, seq, frame) == "accepted"
    with pytest.raises(BacklogFullError):
        engine.ingest(MAX_PENDING_FRAMES, frame)
    # Existing entries and the play cursor are untouched by the refusal.
    state = engine.snapshot()
    assert state["pending_frames"] == MAX_PENDING_FRAMES
    assert state["next_seq"] == 0
    assert engine.playout().seq == 0
    assert engine.ingest(MAX_PENDING_FRAMES, frame).status == "accepted"
