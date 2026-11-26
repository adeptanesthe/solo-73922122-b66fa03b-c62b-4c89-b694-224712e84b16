"""HTTP contract tests, including the demo's loss/reordering scenario."""

from __future__ import annotations

import base64
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.constants import FRAME_BYTES, FRAME_SAMPLES
from app.main import create_app
from helpers import tone_frames


@pytest.fixture()
def client():
    with TestClient(create_app()) as test_client:
        yield test_client


def post_frame(client: TestClient, seq: int, frame: np.ndarray):
    return client.post(
        f"/frames?seq={seq}",
        content=frame.astype("<i2", copy=False).tobytes(),
        headers={"Content-Type": "application/octet-stream"},
    )


def get_frame(client: TestClient):
    response = client.get("/playout")
    assert response.status_code == 200
    samples = np.frombuffer(response.content, dtype="<i2").copy()
    return response, samples


def test_config_advertises_contract(client):
    body = client.get("/config").json()
    assert body["sample_rate_hz"] == 16_000
    assert body["channels"] == 1
    assert body["frame_ms"] == 20
    assert body["frame_samples"] == 320
    assert body["frame_bytes"] == 640
    assert body["period_search_samples"] == [40, 200]
    assert body["min_peak_correlation"] == 0.6


def test_wrong_body_size_is_rejected(client):
    response = client.post(
        "/frames?seq=0",
        content=b"\x00" * (FRAME_BYTES - 2),
        headers={"Content-Type": "application/octet-stream"},
    )
    assert response.status_code == 400
    assert "640" in response.json()["detail"]

    too_big = client.post(
        "/frames?seq=0",
        content=b"\x00" * (FRAME_BYTES + 2),
        headers={"Content-Type": "application/octet-stream"},
    )
    assert too_big.status_code == 413
    assert client.get("/status").json()["pending_frames"] == 0


def test_negative_sequence_is_rejected(client):
    response = post_frame(client, -1, np.zeros(FRAME_SAMPLES, dtype=np.int16))
    assert response.status_code == 422


def test_playout_binary_contract_and_headers(client):
    frame = tone_frames(1)[0]
    assert post_frame(client, 0, frame).json()["status"] == "accepted"

    response, samples = get_frame(client)
    assert response.headers["content-type"] == "application/octet-stream"
    assert len(response.content) == FRAME_BYTES
    assert response.headers["x-sequence"] == "0"
    assert response.headers["x-frame-kind"] == "true"
    assert response.headers["x-next-sequence"] == "1"
    assert response.headers["x-history-samples"] == str(FRAME_SAMPLES)
    np.testing.assert_array_equal(samples, frame)


def test_playout_json_contract(client):
    frame = tone_frames(1)[0]
    post_frame(client, 0, frame)
    body = client.get("/playout?format=json").json()
    assert body["seq"] == 0
    assert body["kind"] == "true"
    assert body["samples"] == FRAME_SAMPLES
    decoded = np.frombuffer(base64.b64decode(body["pcm_b64"]), dtype="<i2")
    np.testing.assert_array_equal(decoded, frame)
    # POST /playout is an alias for clients that dislike mutating GETs.
    assert client.post("/playout").status_code == 200


def test_unknown_format_is_rejected_without_consuming_a_frame(client):
    post_frame(client, 0, tone_frames(1)[0])
    assert client.get("/playout?format=wav").status_code == 400
    # The rejected request must not have pulled seq 0.
    assert client.get("/status").json()["next_seq"] == 0
    assert client.get("/playout").headers["x-sequence"] == "0"


def test_demo_loss_and_reordering_over_http(client):
    frames = tone_frames(8)
    # Arrival order: 0, 1, then 3 and 4 out of order, 2 missing, 6 early.
    for seq in (0, 1, 4, 3, 6):
        assert post_frame(client, seq, frames[seq]).json()["status"] == "accepted"
    assert post_frame(client, 3, frames[3]).json()["status"] == "duplicate"

    kinds = []
    seqs = []
    for _ in range(7):
        response, samples = get_frame(client)
        assert len(samples) == FRAME_SAMPLES
        kinds.append(response.headers["x-frame-kind"])
        seqs.append(int(response.headers["x-sequence"]))
    assert seqs == list(range(7))
    # 2 and 5 are missing; 3 is the first real frame after the gap.
    assert kinds == [
        "true",
        "true",
        "concealed",
        "recovered",
        "true",
        "concealed",
        "recovered",
    ]

    status = client.get("/status").json()
    assert status["next_seq"] == 7
    assert status["pending_seqs"] == []
    assert status["stats"]["true_frames"] == 3
    assert status["stats"]["recovered_frames"] == 2


def test_late_packet_is_ignored_over_http(client):
    frames = tone_frames(5)
    post_frame(client, 0, frames[0])
    post_frame(client, 1, frames[1])
    for _ in range(3):
        get_frame(client)  # seq 2 is concealed

    late = post_frame(client, 2, frames[2])
    assert late.json()["status"] == "late_ignored"

    post_frame(client, 3, frames[3])
    response, samples = get_frame(client)
    assert response.headers["x-sequence"] == "3"
    assert response.headers["x-frame-kind"] == "recovered"
    np.testing.assert_array_equal(samples[80:], frames[3][80:])


def test_concurrent_pulls_each_get_a_distinct_http_frame(client):
    frames = tone_frames(40)
    for seq, frame in enumerate(frames):
        post_frame(client, seq, frame)

    with ThreadPoolExecutor(max_workers=10) as pool:
        responses = list(pool.map(lambda _: client.get("/playout"), range(10)))

    seqs = sorted(int(r.headers["x-sequence"]) for r in responses)
    assert seqs == list(range(10))
    for response in responses:
        seq = int(response.headers["x-sequence"])
        np.testing.assert_array_equal(
            np.frombuffer(response.content, dtype="<i2"), frames[seq]
        )


def test_backlog_limit_returns_429(client, monkeypatch):
    monkeypatch.setattr("app.engine.MAX_PENDING_FRAMES", 2)
    frames = tone_frames(3)
    assert post_frame(client, 0, frames[0]).status_code == 202
    assert post_frame(client, 1, frames[1]).status_code == 202
    response = post_frame(client, 2, frames[2])
    assert response.status_code == 429
    assert client.get("/status").json()["pending_frames"] == 2


def test_status_and_reset(client):
    post_frame(client, 0, tone_frames(1)[0])
    get_frame(client)
    status = client.get("/status?include_history=true").json()
    assert status["next_seq"] == 1
    assert status["history_samples"] == FRAME_SAMPLES
    assert len(base64.b64decode(status["history_pcm_b64"])) == FRAME_BYTES

    assert client.post("/reset").json() == {"status": "reset"}
    after = client.get("/status").json()
    assert after["next_seq"] == 0
    assert after["history_samples"] == 0
