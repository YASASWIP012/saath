"""Loading the C-SSRS labelled posts.

The source CSV carries Reddit usernames and post URLs. We drop both the moment
the file is read, so nothing downstream (features, artifacts, the demo) can
ever surface who wrote a post.
"""

from __future__ import annotations

from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import pandas as pd

SOURCE_URL = (
    "https://huggingface.co/datasets/av9ash/CSSR-S_labelled_suicidewatch_posts_reddit"
    "/resolve/main/labeled_rSuicidewatch_posts.csv"
)
CACHE = Path(__file__).resolve().parent.parent / "data" / "cache"

# Level meanings follow the Columbia Suicide Severity Rating Scale screener,
# using the exact wording the dataset authors gave their annotators.
LEVELS = {
    0: "Not related to suicide",
    1: "Wish to be dead",
    2: "Active suicidal thoughts",
    3: "Thinking about method",
    4: "Intent to act",
    5: "Plan with intent",
    6: "Preparatory act or attempt",
}
N_LEVELS = len(LEVELS)

# The C-SSRS screener treats a yes on intent, plan or recent behaviour as high
# risk. Everything at or above this level must reach a counsellor first.
HIGH_RISK = 4

# Zero-shot labels the dataset authors collected from five frontier LLMs.
JURY = ["gpt", "claude", "gemini", "llama", "mistral"]


def load(path: Path | None = None) -> pd.DataFrame:
    """Return one row per post: text, human level, and the five jury labels."""
    if path is None:
        CACHE.mkdir(parents=True, exist_ok=True)
        path = CACHE / "cssrs.csv"
        if not path.exists():
            urlretrieve(SOURCE_URL, path)

    raw = pd.read_csv(path)
    df = pd.DataFrame(
        {
            "text": raw["content"].astype(str).str.strip(),
            "level": raw["severity"].astype(int),
        }
    )
    for name in JURY:
        df[f"jury_{name}"] = pd.to_numeric(raw[f"{name}_label"], errors="coerce")

    df = df[df.text.str.len() > 0].drop_duplicates("text").reset_index(drop=True)
    df.index.name = "post_id"
    return df


def jury_matrix(df: pd.DataFrame) -> np.ndarray:
    """(n_posts, 5) array of LLM labels, NaN where a model returned nothing."""
    return df[[f"jury_{n}" for n in JURY]].to_numpy(dtype=float)
