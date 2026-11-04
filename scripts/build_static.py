"""Build the static copy of the console that the hosted demo serves.

    uv run python scripts/build_static.py site/

Every JSON file is a response from the real API, recorded ahead of time,
so the hosted page shows exactly what the local server would.
"""

from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.main import app

EXAMPLES = [
    "I keep telling myself it will get better but I've started looking at the pills in the cupboard every night.",
    "I've written the notes and I know which bridge. Tonight after everyone is asleep.",
    "Exams are next week and I can't sleep, I just feel so tired of everything lately.",
    "yaar mujhe ab jeene ka mann nahi karta, sab khatam karna chahta hoon",
    "naaku inka bathakalani ledu ra, anni aapeyalanipistundi",
    "Can you tell me the timings for the counselling centre near Ameerpet?",
]


def main():
    out = Path(sys.argv[1])
    data = out / "data"
    shutil.rmtree(out, ignore_errors=True)
    data.mkdir(parents=True)
    with TestClient(app) as c:
        for name in ["meta", "demo-queue", "results"]:
            (data / f"{name}.json").write_text(c.get(f"/api/{name}").content.decode())
        demo = c.get("/api/demo-queue").json()
        texts = list(dict.fromkeys([d["text"] for d in demo] + EXAMPLES))
        scores = {t: c.post("/api/score", json={"text": t}).json() for t in texts}
    (data / "scores.json").write_text(json.dumps(scores))
    (data / "examples.json").write_text(json.dumps(EXAMPLES))

    static = ROOT / "app" / "static"
    html = (static / "index.html").read_text().replace('"/static/', '"')
    html = re.sub(r"(<script src=)", r"<script>window.SAATH_STATIC = true;</script>\n  \1", html, count=1)
    (out / "index.html").write_text(html)
    for f in ["app.js", "styles.css"]:
        shutil.copy(static / f, out / f)
    print(f"wrote {out} with {len(scores)} scored messages")


if __name__ == "__main__":
    main()
