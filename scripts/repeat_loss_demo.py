#!/usr/bin/env python3
"""API demo focused on burst loss and re-loss after recovery.

Complements ``demo.py`` (reordering / duplicates / late packets) by
exercising the period estimator's incremental path end to end over HTTP:

* burst A: three consecutive missing frames, then recovery with the 5 ms
  fade, then plain true frames;
* burst B: another loss run *after* the stream has fully recovered, proving
  the estimator re-engages on windows shifted by more than one frame;
* a single isolated loss between healthy frames.

The script only uses the Python standard library.  It verifies frame kinds,
playout order, selected pitch periods and peak headers, the untouched true
samples after each fade, and the recursive amplitude halving inside a burst,
then writes ``<out>/repeat_loss_transcript.json`` and ``<out>/repeat_loss.wav``.

Example:
    python3 scripts/repeat_loss_demo.py --url http://127.0.0.1:8000 --out demo_out
"""

from __future__ import annotations

import argparse
import json
import math
import os
import struct
import sys
import urllib.error
import urllib.request
import wave
from pathlib import Path

SAMPLE_RATE = 16_000
FRAME_SAMPLES = 320
FRAME_BYTES = FRAME_SAMPLES * 2
FADE_SAMPLES = 80
PERIOD = 160  # 100 Hz fundamental -> detected pitch lag of 160 samples


def _header(headers: dict, name: str) -> str:
    value = headers.get(name)
    if value is None:
        value = headers.get(name.lower(), "")
    return value


def _request(url: str, method: str = "GET", body: bytes | None = None):
    headers = {"Content-Type": "application/octet-stream"} if body is not None else {}
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            return response.status, response.read(), dict(response.headers)
    except urllib.error.HTTPError as error:
        return error.code, error.read(), dict(error.headers)


def wait_for_health(base_url: str, timeout: float = 60.0) -> None:
    import time

    deadline = time.monotonic() + timeout
    while True:
        try:
            status, _, _ = _request(f"{base_url}/healthz")
            if status == 200:
                return
        except OSError:
            pass
        if time.monotonic() > deadline:
            raise SystemExit(f"API did not become healthy at {base_url}")
        time.sleep(0.5)


def make_frames(count: int, amplitude: float = 11_000.0) -> list[bytes]:
    """One phase-continuous 100 Hz harmonic tone split into PCM16 frames."""

    total = count * FRAME_SAMPLES
    samples = []
    for n in range(total):
        t = n / SAMPLE_RATE
        envelope = 0.65 + 0.35 * math.sin(2 * math.pi * 2.0 * t)
        value = (
            0.62 * math.sin(2 * math.pi * n / PERIOD)
            + 0.27 * math.sin(2 * math.pi * 2 * n / PERIOD + 0.4)
            + 0.11 * math.sin(2 * math.pi * 3 * n / PERIOD + 1.1)
        )
        samples.append(int(round(amplitude * envelope * value)))
    pcm = b"".join(struct.pack("<h", max(-32768, min(32767, s))) for s in samples)
    return [pcm[i * FRAME_BYTES : (i + 1) * FRAME_BYTES] for i in range(count)]


# Sequences 4,5,6 form the first burst; 11,12 the second burst after
# recovery; 15 is an isolated loss.  Everything else is delivered.
FIRST_BURST = (4, 5, 6)
SECOND_BURST = (11, 12)
ISOLATED_LOSS = (15,)
LOST = set(FIRST_BURST) | set(SECOND_BURST) | set(ISOLATED_LOSS)
TOTAL_FRAMES = 20


class RepeatLossDemo:
    def __init__(self, base_url: str, out_dir: Path) -> None:
        self.base_url = base_url.rstrip("/")
        self.out_dir = out_dir
        self.frames = make_frames(TOTAL_FRAMES)
        self.played: list[bytes] = []
        self.kinds: list[str] = []
        self.seqs: list[int] = []
        self.periods: list[int | None] = []
        self.transcript: list[dict] = []

    def post(self, seq: int) -> None:
        status, body, _ = _request(
            f"{self.base_url}/frames?seq={seq}", "POST", self.frames[seq]
        )
        if status != 202:
            raise SystemExit(f"POST seq={seq} failed with {status}: {body!r}")
        detail = json.loads(body)["status"]
        self.transcript.append({"action": "post", "seq": seq, "result": detail})

    def pull(self, note: str) -> None:
        status, body, headers = _request(f"{self.base_url}/playout")
        if status != 200:
            raise SystemExit(f"GET /playout failed with {status}: {body!r}")
        if len(body) != FRAME_BYTES:
            raise SystemExit(f"unexpected playout size {len(body)}")
        seq = int(_header(headers, "x-sequence"))
        kind = _header(headers, "x-frame-kind")
        period_raw = _header(headers, "x-period-samples")
        period = int(period_raw) if period_raw else None
        peak = _header(headers, "x-peak-correlation")
        self.played.append(body)
        self.seqs.append(seq)
        self.kinds.append(kind)
        self.periods.append(period)
        self.transcript.append(
            {
                "action": "pull",
                "seq": seq,
                "result": kind,
                "period_samples": period,
                "peak_correlation": float(peak) if peak else None,
                "note": note,
            }
        )
        line = f"PULL seq={seq:<2d} {kind:<10s}"
        if period is not None:
            line += f" period={period} peak={float(peak):.4f}"
        line += f"  ({note})"
        print(line)

    def run(self) -> None:
        wait_for_health(self.base_url)
        _request(f"{self.base_url}/reset", "POST")

        print("== prefetch the healthy prefix (0..3), losses 4,5,6 withheld ==")
        for seq in range(4):
            self.post(seq)
        for seq in range(4):
            self.pull("healthy true frame")

        print("== burst A: three consecutive losses (4, 5, 6) ==")
        self.post(7)
        for _ in range(4):
            self.pull("concealment burst then recovery pull")

        print("== recovery settles: true frames 7..10, then burst B (11,12) ==")
        for seq in (8, 9, 10):
            self.post(seq)
        for seq in (8, 9, 10):
            self.pull("true frame after recovery")
        self.post(13)
        for _ in range(3):
            self.pull("loss run after recovery")

        print("== isolated loss at seq 15 between healthy frames ==")
        for seq in (14, 16, 17, 18, 19):
            self.post(seq)
        for _ in range(6):
            self.pull("isolated loss then healthy tail")

        assert len(self.played) == TOTAL_FRAMES
        status_code, status_body, _ = _request(f"{self.base_url}/status")
        if status_code != 200:
            raise SystemExit(f"GET /status failed with {status_code}")
        status = json.loads(status_body)
        self._verify(status)
        self._write_outputs(status)

    def _verify(self, status: dict) -> None:
        problems: list[str] = []

        expected_kinds = []
        recovered_seqs = set()
        for seq in range(TOTAL_FRAMES):
            if seq in LOST:
                expected_kinds.append("concealed")
            elif seq - 1 in LOST:
                expected_kinds.append("recovered")
                recovered_seqs.add(seq)
            else:
                expected_kinds.append("true")

        if self.seqs != list(range(TOTAL_FRAMES)):
            problems.append(f"playout order {self.seqs} is not 0..{TOTAL_FRAMES - 1}")
        if self.kinds != expected_kinds:
            problems.append(f"frame kinds mismatch:\n  got {self.kinds}\n  exp {expected_kinds}")

        # Every concealment on this periodic tone must select the same
        # fundamental period regardless of which cache path produced it.
        for seq in sorted(LOST):
            if self.periods[seq] != PERIOD:
                problems.append(
                    f"seq {seq}: expected period {PERIOD}, got {self.periods[seq]}"
                )

        def pcm(index: int) -> list[int]:
            return list(struct.unpack(f"<{FRAME_SAMPLES}h", self.played[index]))

        # Recovered frames: the fade only touches the first 80 samples.
        for seq in sorted(recovered_seqs):
            true = struct.unpack(f"<{FRAME_SAMPLES}h", self.frames[seq])
            if pcm(seq)[FADE_SAMPLES:] != list(true[FADE_SAMPLES:]):
                problems.append(f"seq {seq}: samples after fade were altered")

        # Inside the three-frame burst, each successive concealment halves the
        # peak level because the tiled period is itself a halved concealment.
        levels = [max(abs(v) for v in pcm(seq)) for seq in FIRST_BURST]
        if levels[0] == 0:
            problems.append("first burst concealment was silence on a periodic tone")
        for prev, nxt in zip(levels, levels[1:]):
            if abs(nxt - prev / 2) > prev * 0.1:
                problems.append(
                    f"burst levels did not halve: {prev} -> {nxt}"
                )

        stats = status["stats"]
        if stats["concealed_frames"] != len(LOST):
            problems.append(
                f"expected {len(LOST)} concealed frames, got {stats['concealed_frames']}"
            )
        if stats["recovered_frames"] != 3:
            problems.append(f"expected 3 recovered frames, got {stats['recovered_frames']}")
        if status["history_samples"] > 640:
            problems.append("history exceeded the 40 ms bound")

        if problems:
            raise SystemExit(
                "repeat-loss demo verification failed:\n  " + "\n  ".join(problems)
            )
        print("\nverification: OK")

    def _write_outputs(self, status: dict) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        wav_path = self.out_dir / "repeat_loss.wav"
        with wave.open(str(wav_path), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(SAMPLE_RATE)
            wav.writeframes(b"".join(self.played))

        transcript_path = self.out_dir / "repeat_loss_transcript.json"
        transcript_path.write_text(
            json.dumps(
                {
                    "lost_sequences": sorted(LOST),
                    "frames": self.transcript,
                    "status": status,
                },
                indent=2,
            )
            + "\n"
        )
        counts: dict[str, int] = {}
        for kind in self.kinds:
            counts[kind] = counts.get(kind, 0) + 1
        print("\n== summary ==")
        print(f"pulled frames: {len(self.played)}  by kind: {counts}")
        print(f"engine stats:  {status['stats']}")
        print(f"wrote {wav_path} and {transcript_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--url",
        default=os.environ.get("MONITOR_URL", "http://127.0.0.1:8000"),
        help="base URL of the monitor-return API",
    )
    parser.add_argument(
        "--out",
        default=os.environ.get("MONITOR_DEMO_OUT", "demo_out"),
        help="directory for wav and transcript outputs",
    )
    args = parser.parse_args()
    RepeatLossDemo(args.url, Path(args.out)).run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
