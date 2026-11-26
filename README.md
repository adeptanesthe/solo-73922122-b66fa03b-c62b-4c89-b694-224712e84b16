# Monitor Return

Single-channel monitor-return ("返听") service for a live mixing desk. The
engine plays numbered 20 ms frames strictly in order from sequence 0 and fills
missing frames from the pitch period of the most recent 40 ms of audio that was
actually played, so the engineer keeps hearing the dialogue instead of muting.

## Audio contract

| Item | Value |
| --- | --- |
| Sample rate | 16 000 Hz |
| Channels | 1 (mono) |
| Sample format | little-endian signed 16-bit PCM (`pcm_s16le`) |
| Frame | 20 ms = 320 samples = 640 bytes |
| Sequence | non-negative integer, starts at 0, may arrive in any order |
| History for concealment | last 640 samples (40 ms) that were actually emitted |
| Pitch search | lags 40..200 samples (80..400 Hz), inclusive |
| Correlation gate | peak normalised correlation must be >= 0.6 |
| Concealment | tile the last detected period over one frame at 0.5 gain |
| Recovery | first 5 ms (80 samples) of the first real frame after a gap is a linear blend from the hypothetical concealment; everything else is returned verbatim |

## HTTP API

| Method | Path | Meaning |
| --- | --- | --- |
| `POST` | `/frames?seq=N` | Raw 640-byte PCM16 frame. `202` with `accepted`, `duplicate` or `late_ignored`. Wrong size `400`, oversized `413`, backlog full `429`. |
| `GET`/`POST` | `/playout` | Pull exactly one frame. Binary body (640 bytes) plus `X-*` metadata headers. `?format=json` returns base64 samples and metadata. |
| `GET` | `/status` | Cursor, pending sequence numbers, counters, optional played history (`?include_history=true`). |
| `POST` | `/reset` | Clear all state (used by the demo). |
| `GET` | `/config` | The format/concealment parameters above. |
| `GET` | `/healthz` | Liveness for Docker. |

Playout metadata headers: `X-Sequence`, `X-Next-Sequence`, `X-Frame-Kind`
(`true` / `recovered` / `concealed` / `silence`), `X-Period-Samples`,
`X-Peak-Correlation`, `X-Conceal-Reason` (`periodic` / `insufficient_history` /
`autocorr_undefined` / `low_correlation`), `X-Crossfade-Samples`,
`X-History-Samples`.

## Semantics

* **Ordering.** The playout cursor only moves forward from 0. A missing
  sequence number is concealed once its turn is pulled; later numbers wait.
* **Concealment.** Mean-removed, per-lag normalised autocorrelation over the
  last 40 ms of *played* audio. The highest peak in lags 40..200 is the
  period. The last period is tiled over 320 samples at half amplitude.
  Shortest lag wins within 1e-9 of the maximum, so a periodic signal reports
  its fundamental rather than a multiple.
* **Consecutive loss.** Each concealed frame is appended to the history, so the
  next missing frame recursively repeats (and halves) the newest period.
* **Silence.** Fewer than 40 ms of history, an undefined autocorrelation (all
  samples equal or zero energy), or a peak below 0.6 produces a silent frame.
* **Recovery.** The first real frame after one or more missing frames is blended
  over its first 80 samples from what concealment *would* have produced for
  that slot, then returns the true samples untouched. Following frames are
  returned verbatim (and also fade in after a gap whose concealment was
  silence).
* **Duplicates.** The first copy of a sequence number is kept; later copies are
  acknowledged as `duplicate` and discarded.
* **Late packets.** Once a sequence number's turn has been pulled, any packet
  for it is `late_ignored`. It cannot change the emitted audio or the played
  history that later concealment is derived from.
* **Concurrency.** Ingestion and playout use separate locks for bookkeeping;
  NumPy work runs outside the state lock, so ``/frames`` keeps accepting while
  a frame is being concealed. Pulls are serialised, so concurrent pulls each
  return exactly one distinct, sequential frame.
* **Backlog bound.** At most 5 000 pending frames (100 s). Beyond that new
  future frames get `429`; nothing accepted is evicted and the cursor does not
  move.

## Run with Docker Compose

```bash
docker compose up --build
# in another shell
python3 scripts/demo.py --url http://127.0.0.1:8000 --out demo_out

# or run the same demo as a one-shot compose service
docker compose --profile demo run --rm demo
```

The demo only needs the Python standard library. It plays a 100 Hz harmonic
test signal, delivers frames out of order with one duplicate and two late
retransmissions, loses sequences 2, 6, 9 and 10, pulls while frames are still
arriving, then pulls four frames concurrently. It checks the playout order,
frame kinds, the untouched samples after each 5 ms fade and the halving between
consecutive concealments. It writes `demo_out/playout.wav` (listenable) and
`demo_out/transcript.json`, and exits non-zero if a check fails.

Raw curl equivalent of the key cases:

```bash
# frame N is the 640-byte body; generate with scripts/demo.py's make_frames
curl -sS -X POST --data-binary @f000.bin -H 'Content-Type: application/octet-stream' \
  'http://127.0.0.1:8000/frames?seq=0'
curl -sS -X POST --data-binary @f001.bin -H 'Content-Type: application/octet-stream' \
  'http://127.0.0.1:8000/frames?seq=1'
curl -sS -X POST --data-binary @f003.bin -H 'Content-Type: application/octet-stream' \
  'http://127.0.0.1:8000/frames?seq=3'        # 2 is lost, 3 arrives early
curl -sS -D - -o out0.pcm 'http://127.0.0.1:8000/playout'   # X-Frame-Kind: true
curl -sS -D - -o out1.pcm 'http://127.0.0.1:8000/playout'   # true
curl -sS -D - -o out2.pcm 'http://127.0.0.1:8000/playout'   # concealed
curl -sS -D - -o out3.pcm 'http://127.0.0.1:8000/playout'   # recovered
curl -sS -X POST --data-binary @f002.bin -H 'Content-Type: application/octet-stream' \
  'http://127.0.0.1:8000/frames?seq=2'        # late_ignored
```

## Local development

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements-dev.txt
pytest                                   # 41 tests
uvicorn app.main:app --port 8000         # then run scripts/demo.py
```

Playback state lives in process memory, so run **one** uvicorn worker (the
Docker image does). Running several workers or replicas would give each its own
cursor; that needs sticky routing or an external shared state store.

Layout: `app/concealment.py` (NumPy period search, concealment, fade),
`app/engine.py` (ordered playout state machine), `app/main.py` (FastAPI),
`scripts/demo.py` (lossy/reordered demo + self-check), `tests/`.
