"""FastAPI surface for the single-channel monitor return.

Endpoints
---------
POST /frames?seq=N
    Raw body of exactly 640 bytes: 320 little-endian int16 samples (20 ms).
    Frames may arrive in any order.  A duplicate ``seq`` keeps the first copy;
    a ``seq`` whose playback turn has already occurred is ignored.

GET /playout
    Returns exactly one frame and advances the cursor.  The default body is
    320 little-endian int16 samples; ``?format=json`` returns base64 samples
    plus the same metadata for inspection.

GET /status, POST /reset
    Diagnostics and a full state reset.
"""

from __future__ import annotations

import base64
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException, Query, Request, Response

from .concealment import FADE_SAMPLES
from .constants import (
    CONCEAL_GAIN,
    FRAME_BYTES,
    FRAME_MS,
    FRAME_SAMPLES,
    HISTORY_MS,
    HISTORY_SAMPLES,
    MAX_PERIOD_SAMPLES,
    MIN_PEAK_CORRELATION,
    MIN_PERIOD_SAMPLES,
    SAMPLE_RATE,
)
from .engine import BacklogFullError, PlayoutEngine

_PCM = "application/octet-stream"


async def _read_frame_body(request: Request) -> bytes:
    """Read the raw frame body, refusing anything that is not one frame.

    The body is streamed with a hard cap so an oversized upload is rejected
    as soon as it crosses one frame instead of being buffered in full.
    """

    chunks: list[bytes] = []
    size = 0
    async for chunk in request.stream():
        size += len(chunk)
        if size > FRAME_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"frame body exceeds {FRAME_BYTES} bytes",
            )
        chunks.append(chunk)
    payload = b"".join(chunks)
    if len(payload) != FRAME_BYTES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"expected {FRAME_BYTES} bytes for one "
                f"{FRAME_MS} ms mono PCM16 frame, got {len(payload)}"
            ),
        )
    return payload


def _format_config() -> dict:
    return {
        "sample_rate_hz": SAMPLE_RATE,
        "channels": 1,
        "sample_format": "pcm_s16le",
        "frame_ms": FRAME_MS,
        "frame_samples": FRAME_SAMPLES,
        "frame_bytes": FRAME_BYTES,
        "history_ms": HISTORY_MS,
        "history_samples": HISTORY_SAMPLES,
        "period_search_samples": [MIN_PERIOD_SAMPLES, MAX_PERIOD_SAMPLES],
        "min_peak_correlation": MIN_PEAK_CORRELATION,
        "conceal_gain": CONCEAL_GAIN,
        "recovery_fade_ms": 5,
        "recovery_fade_samples": FADE_SAMPLES,
    }


def _playout_headers(result) -> dict[str, str]:
    return {
        "X-Sequence": str(result.seq),
        "X-Next-Sequence": str(result.next_seq),
        "X-Frame-Kind": result.kind,
        "X-Period-Samples": (
            "" if result.period_samples is None else str(result.period_samples)
        ),
        "X-Peak-Correlation": (
            "" if result.peak is None else f"{result.peak:.6f}"
        ),
        "X-Conceal-Reason": result.conceal_reason or "",
        "X-Crossfade-Samples": str(result.crossfade_samples),
        "X-History-Samples": str(result.history_samples),
    }


def _playout_json(result) -> dict:
    payload = {
        "seq": result.seq,
        "kind": result.kind,
        "samples": int(result.samples.size),
        "sample_format": "pcm_s16le",
        "pcm_b64": base64.b64encode(
            result.samples.astype("<i2", copy=False).tobytes()
        ).decode("ascii"),
        "period_samples": result.period_samples,
        "peak_correlation": result.peak,
        "conceal_reason": result.conceal_reason,
        "crossfade_samples": result.crossfade_samples,
        "history_samples": result.history_samples,
        "next_seq": result.next_seq,
    }
    return payload


def create_app() -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.engine = PlayoutEngine()
        yield

    app = FastAPI(
        title="Monitor Return",
        version="1.0.0",
        description="Ordered 16 kHz monitor return with period-based concealment.",
        lifespan=lifespan,
    )

    def engine(request: Request) -> PlayoutEngine:
        return request.app.state.engine

    @app.get("/healthz")
    def healthz() -> dict:
        return {"ok": True}

    @app.get("/config")
    def config() -> dict:
        return _format_config()

    @app.post("/frames", status_code=202)
    async def ingest_frame(
        request: Request,
        seq: int = Query(..., ge=0),
    ) -> dict:
        payload = await _read_frame_body(request)
        samples = np.frombuffer(payload, dtype="<i2").astype(np.int16, copy=True)
        try:
            result = engine(request).ingest(seq, samples)
        except BacklogFullError as error:
            raise HTTPException(status_code=429, detail=str(error)) from error
        return {
            "seq": result.seq,
            "status": result.status,
            "pending": result.pending,
            "next_seq": result.next_seq,
        }

    @app.get("/playout")
    @app.post("/playout")
    def playout(request: Request, format: str = Query("binary")):
        if format not in ("binary", "json"):
            raise HTTPException(
                status_code=400, detail="format must be 'binary' or 'json'"
            )
        result = engine(request).playout()
        if format == "json":
            return _playout_json(result)
        return Response(
            content=result.samples.astype("<i2", copy=False).tobytes(),
            media_type=_PCM,
            headers=_playout_headers(result),
        )

    @app.get("/status")
    def status(request: Request, include_history: bool = Query(False)) -> dict:
        state = engine(request).snapshot()
        body = {
            "next_seq": state["next_seq"],
            "pending_frames": state["pending_frames"],
            "pending_seqs": state["pending_seqs"],
            "history_samples": state["history_samples"],
            "had_gap": state["had_gap"],
            "stats": state["stats"],
            "config": _format_config(),
        }
        if include_history:
            body["history_pcm_b64"] = base64.b64encode(
                state["history"].astype("<i2", copy=False).tobytes()
            ).decode("ascii")
        return body

    @app.post("/reset")
    def reset(request: Request) -> dict:
        engine(request).reset()
        return {"status": "reset"}

    return app


app = create_app()
