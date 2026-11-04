"""API contract, using a small TF-IDF model so the test needs no downloads."""

import importlib
import json
import os

import joblib
import numpy as np
import pytest
from fastapi.testclient import TestClient

from saath import models
from saath.service import Calibration

TEXTS = [
    "had a lovely day with friends",
    "exam stress is getting to me",
    "I wish I could sleep and not wake up",
    "I keep thinking about killing myself",
    "I have been looking at the pills in the cupboard",
    "I am going to do it this weekend, I've decided",
    "I wrote the letters and picked saturday",
    "I already took the whole bottle",
] * 6


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    art = tmp_path_factory.mktemp("artifacts")
    y = np.tile([0, 1, 2, 3, 4, 5, 6, 6], 6)
    model = models.TfidfOrdinal(C=4.0).fit(TEXTS, y)
    joblib.dump(model, art / "model.joblib")
    Calibration.fit(model.exceed_proba(TEXTS), y).save(art / "calibration.json")
    (art / "model.json").write_text(json.dumps({"kind": "sklearn"}))
    os.environ["SAATH_ARTIFACTS"] = str(art)
    from app import main

    importlib.reload(main)
    with TestClient(main.app) as c:
        yield c


def test_score_returns_a_consistent_record(client):
    r = client.post("/api/score", json={"text": "I already took the whole bottle. I feel dizzy."})
    assert r.status_code == 200
    s = r.json()
    assert len(s["proba"]) == 7
    assert sum(s["proba"]) == pytest.approx(1, abs=1e-3)
    lo, hi = s["level_range"]
    assert 0 <= lo <= hi <= 6
    assert s["band"] in {"urgent", "review", "routine"}
    assert 0 <= s["risk_calibrated"] <= 1
    assert [e["text"] for e in s["evidence"]] == ["I already took the whole bottle.", "I feel dizzy."]


def test_romanised_indic_is_never_routine(client):
    s = client.post("/api/score", json={"text": "aaj bahut accha din tha yaar, sab theek hai"}).json()
    assert s["band"] != "routine"
    assert s["band_reason"] in {"language", "risk", "range"}


def test_empty_and_oversized_messages_are_rejected(client):
    assert client.post("/api/score", json={"text": ""}).status_code == 422
    assert client.post("/api/score", json={"text": "x" * 5000}).status_code == 422


def test_batch_matches_single(client):
    texts = ["exam stress is getting to me", "I already took the whole bottle"]
    batch = client.post("/api/score-batch", json={"texts": texts}).json()
    for t, b in zip(texts, batch, strict=True):
        single = client.post("/api/score", json={"text": t}).json()
        assert single["band"] == b["band"]
        assert single["risk"] == pytest.approx(b["risk"], abs=1e-4)
