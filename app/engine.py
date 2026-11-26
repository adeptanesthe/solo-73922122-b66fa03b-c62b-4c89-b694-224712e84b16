"""Ordered playout state machine with immutable played history.

Concurrency model
-----------------
* ``_play_lock`` serialises pulls.  Each successful pull takes exactly one
  sequence number and returns exactly one frame.
* ``_state_lock`` protects the pending map, the sequence counter, the played
  history and statistics.  It is held only for short bookkeeping sections,
  never while NumPy concealment runs.  Ingestion therefore keeps working
  while a frame is being concealed.
* The history is built exclusively from frames that were actually emitted.
  A late packet is rejected once its sequence number has been assigned, so it
  can neither replace a played frame nor change the history used by later
  concealment.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

import numpy as np

from .concealment import Concealment, concealment_for, crossfade_to_true
from .constants import (
    FADE_SAMPLES,
    FRAME_SAMPLES,
    HISTORY_SAMPLES,
    MAX_PENDING_FRAMES,
)


class BacklogFullError(RuntimeError):
    """Raised when the reordering buffer cannot accept more future frames."""


@dataclass(frozen=True)
class IngestResult:
    seq: int
    status: str  # "accepted", "duplicate", "late_ignored"
    pending: int
    next_seq: int


@dataclass(frozen=True)
class PlayoutResult:
    seq: int
    kind: str  # "true", "recovered", "concealed", "silence"
    samples: np.ndarray
    period_samples: int | None
    peak: float | None
    conceal_reason: str | None
    crossfade_samples: int
    history_samples: int
    next_seq: int


class PlayoutEngine:
    """Single-channel, sequence-addressed monitor return."""

    def __init__(self) -> None:
        self._state_lock = threading.Lock()
        self._play_lock = threading.Lock()
        self._pending: dict[int, np.ndarray] = {}
        self._next_seq = 0
        self._history = np.zeros(0, dtype=np.int16)
        self._had_gap = False
        self._stats = {
            "accepted_frames": 0,
            "duplicate_frames": 0,
            "late_frames": 0,
            "true_frames": 0,
            "recovered_frames": 0,
            "concealed_frames": 0,
            "silence_frames": 0,
        }

    # ------------------------------------------------------------------ input

    def ingest(self, seq: int, samples: np.ndarray) -> IngestResult:
        """Accept one numbered frame.  The first copy of a number wins."""

        if seq < 0:
            raise ValueError("seq must be non-negative")
        frame = np.asarray(samples)
        if frame.ndim != 1 or frame.size != FRAME_SAMPLES:
            raise ValueError(
                f"frame must contain exactly {FRAME_SAMPLES} samples, got {frame.size}"
            )
        if frame.dtype != np.int16:
            raise ValueError("frame must be int16 PCM")
        stored = frame.copy()

        with self._state_lock:
            if seq < self._next_seq:
                # Its playback turn has already happened, possibly as
                # concealment.  Ignore it entirely.
                self._stats["late_frames"] += 1
                return IngestResult(
                    seq=seq,
                    status="late_ignored",
                    pending=len(self._pending),
                    next_seq=self._next_seq,
                )
            if seq in self._pending:
                self._stats["duplicate_frames"] += 1
                return IngestResult(
                    seq=seq,
                    status="duplicate",
                    pending=len(self._pending),
                    next_seq=self._next_seq,
                )
            if len(self._pending) >= MAX_PENDING_FRAMES:
                raise BacklogFullError(
                    f"reordering buffer full ({MAX_PENDING_FRAMES} frames); "
                    "wait for playout to drain it"
                )
            self._pending[seq] = stored
            self._stats["accepted_frames"] += 1
            return IngestResult(
                seq=seq,
                status="accepted",
                pending=len(self._pending),
                next_seq=self._next_seq,
            )

    # ----------------------------------------------------------------- output

    def playout(self) -> PlayoutResult:
        """Return exactly one frame, advancing the playout cursor."""

        with self._play_lock:
            # Commit this turn and snapshot everything the decision needs.
            with self._state_lock:
                seq = self._next_seq
                self._next_seq += 1
                frame = self._pending.pop(seq, None)
                history = self._history.copy()
                had_gap = self._had_gap

            hidden: Concealment | None = None
            fade_samples = 0

            if frame is None:
                hidden = self._concealment_for(history)
                out = hidden.samples
                if hidden.applied:
                    kind = "concealed"
                else:
                    kind = "silence"
                new_had_gap = True
            elif had_gap:
                # The first real frame after a gap: synthesize what would
                # have been played for this slot, then blend into the truth.
                hidden = self._concealment_for(history)
                out = crossfade_to_true(hidden.samples, frame)
                kind = "recovered"
                fade_samples = FADE_SAMPLES
                new_had_gap = False
            else:
                out = frame.copy()
                kind = "true"
                new_had_gap = False

            with self._state_lock:
                self._append_history(out)
                self._had_gap = new_had_gap
                if kind == "true":
                    self._stats["true_frames"] += 1
                elif kind == "recovered":
                    self._stats["recovered_frames"] += 1
                elif kind == "concealed":
                    self._stats["concealed_frames"] += 1
                else:
                    self._stats["silence_frames"] += 1
                history_samples = int(self._history.size)
                next_seq = self._next_seq

            return PlayoutResult(
                seq=seq,
                kind=kind,
                samples=out,
                period_samples=hidden.period_samples if hidden else None,
                peak=hidden.peak if hidden else None,
                conceal_reason=hidden.reason if hidden else None,
                crossfade_samples=fade_samples,
                history_samples=history_samples,
                next_seq=next_seq,
            )

    # ------------------------------------------------------------------ state

    def snapshot(self) -> dict:
        """Return a consistent copy of the engine state for diagnostics."""

        with self._state_lock:
            return {
                "next_seq": self._next_seq,
                "pending_seqs": sorted(self._pending),
                "pending_frames": len(self._pending),
                "history_samples": int(self._history.size),
                "history": self._history.copy(),
                "had_gap": self._had_gap,
                "stats": dict(self._stats),
            }

    def reset(self) -> None:
        """Clear all state.  Waits for an in-flight pull to finish."""

        with self._play_lock:
            with self._state_lock:
                self._pending.clear()
                self._next_seq = 0
                self._history = np.zeros(0, dtype=np.int16)
                self._had_gap = False
                for key in self._stats:
                    self._stats[key] = 0

    # --------------------------------------------------------------- internal

    def _concealment_for(self, history: np.ndarray) -> Concealment:
        """Indirection so ingestion-during-concealment can be exercised."""

        return concealment_for(history)

    def _append_history(self, frame: np.ndarray) -> None:
        if frame.size >= HISTORY_SAMPLES:
            self._history = np.asarray(frame[-HISTORY_SAMPLES:], dtype=np.int16).copy()
            return
        merged = np.concatenate((self._history, np.asarray(frame, dtype=np.int16)))
        self._history = merged[-HISTORY_SAMPLES:].copy()
