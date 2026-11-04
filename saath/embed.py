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


def encode(texts: list[str], model: str, use_cache: bool = True) -> np.ndarray:
    """L2-normalised embeddings, shape (len(texts), dim)."""
    key = hashlib.sha1((model + "\x00" + "\x00".join(texts)).encode()).hexdigest()[:16]
    path = CACHE / f"{model.replace('/', '__')}-{key}.npy"
    if use_cache and path.exists():
        return np.load(path)

    pre = _prefix(model)
    vecs = _encoder(model).encode(
        [pre + t for t in texts],
        batch_size=32,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    vecs = np.asarray(vecs, dtype=np.float32)
    if use_cache:
        CACHE.mkdir(parents=True, exist_ok=True)
        np.save(path, vecs)
    return vecs
