"""Sentence embeddings with an on-disk cache keyed by model and text."""

from __future__ import annotations

import hashlib
import os
from functools import lru_cache
from pathlib import Path

import numpy as np

CACHE = Path(__file__).resolve().parent.parent / "data" / "cache" / "emb"

# e5 models expect a task prefix on every input.
_PREFIX = {"intfloat/": "query: "}


def _prefix(model: str) -> str:
    return next((p for k, p in _PREFIX.items() if model.startswith(k)), "")


_DEVICE = os.environ.get("SAATH_DEVICE")  # unset picks the GPU when one is available


@lru_cache(maxsize=4)
def _encoder(model: str):
    import torch
    from sentence_transformers import SentenceTransformer

    device = _DEVICE or ("mps" if torch.backends.mps.is_available() else "cpu")
    return SentenceTransformer(model, device=device)


_MEMO: dict[str, dict[str, np.ndarray]] = {}


def _store(model: str) -> Path:
    return CACHE / f"{model.replace('/', '__')}.npz"


def _memo(model: str) -> dict[str, np.ndarray]:
    """Per-text embeddings for one model, loaded from disk once."""
    if model not in _MEMO:
        memo = {}
        path = _store(model)
        if path.exists():
            z = np.load(path)
            memo = dict(zip(z["keys"].tolist(), z["vecs"], strict=True))
        _MEMO[model] = memo
    return _MEMO[model]


def _key(text: str) -> str:
    return hashlib.sha1(text.encode()).hexdigest()


def _encode_raw(texts: list[str], model: str) -> np.ndarray:
    pre = _prefix(model)
    vecs = _encoder(model).encode(
        [pre + t for t in texts], batch_size=32, normalize_embeddings=True, show_progress_bar=False
    )
    return np.asarray(vecs, dtype=np.float32)


def encode(texts: list[str], model: str, use_cache: bool = True) -> np.ndarray:
    """L2-normalised embeddings, shape (len(texts), dim).

    With the cache on, each distinct text is encoded once per model and
    reused across every fold and split. With it off (live scoring), nothing
    is kept in memory or written to disk.
    """
    if not use_cache:
        return _encode_raw(texts, model)
    memo = _memo(model)
    keys = [_key(t) for t in texts]
    missing = sorted({k: t for k, t in zip(keys, texts, strict=True) if k not in memo}.items())
    if missing:
        vecs = _encode_raw([t for _, t in missing], model)
        memo.update({k: v for (k, _), v in zip(missing, vecs, strict=True)})
        CACHE.mkdir(parents=True, exist_ok=True)
        np.savez(_store(model), keys=np.array(list(memo)), vecs=np.stack(list(memo.values())))
    return np.stack([memo[k] for k in keys])
