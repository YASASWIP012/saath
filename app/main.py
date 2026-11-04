"""HTTP API and static console.

uv run uvicorn app.main:app --reload
"""

from __future__ import annotations

import json
import os
import sys
import threading
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from saath.data import HIGH_RISK, LEVELS
from saath.service import Calibration, Scorer

ARTIFACTS = Path(os.environ.get("SAATH_ARTIFACTS", ROOT / "artifacts"))
STATIC = Path(__file__).resolve().parent / "static"
RESULTS = ROOT / "results"


@lru_cache(maxsize=1)
def scorer() -> Scorer:
    import joblib

    model = joblib.load(ARTIFACTS / "model.joblib")
    scorer_ = Scorer(model, Calibration.load(ARTIFACTS / "calibration.json"))
    scorer_.score("warm up", explain=False)  # loads both encoders
    return scorer_


@asynccontextmanager
async def lifespan(_app):
    # Load the encoders in the background so the page is up immediately.
    threading.Thread(target=scorer, daemon=True).start()
    yield


app = FastAPI(title="Saath triage API", docs_url="/api/docs", openapi_url="/api/openapi.json", lifespan=lifespan)


class Message(BaseModel):
    text: str = Field(min_length=1, max_length=4000)


class Batch(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=64)


@app.post("/api/score")
def score(msg: Message):
    return scorer().score(msg.text)


@app.post("/api/score-batch")
def score_batch(batch: Batch):
    return scorer().score_many(batch.texts)


@app.get("/api/meta")
def meta():
    cal = scorer().cal
    return {
        "levels": LEVELS,
        "high_risk_level": HIGH_RISK,
        "urgent_threshold": cal.urgent_threshold,
        "alpha_urgent": cal.alpha_urgent,
        "alpha_level": cal.alpha_level,
    }


@app.get("/api/demo-queue")
def demo_queue():
    return json.loads((Path(__file__).resolve().parent / "demo_messages.json").read_text())


@app.get("/api/results")
def results():
    path = RESULTS / "report.json"
    if not path.exists():
        raise HTTPException(404, "run scripts/evaluate.py first")
    return json.loads(path.read_text())


@app.get("/healthz")
def healthz():
    return {"ok": True}


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


app.mount("/static", StaticFiles(directory=STATIC), name="static")
