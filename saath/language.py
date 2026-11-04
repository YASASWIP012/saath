"""Spotting romanised Hindi and Telugu.

Our severity model was trained on English posts, and it reads romanised
Hindi and Telugu badly (see results/codemixed.json). Rather than pretend
otherwise, the service detects these messages and refuses to place them in
the standard ("routine") band. A counsellor sees them flagged as "language: review".

The detector is deliberately simple: high-frequency function words and
pronouns that rarely appear in English. It needs to be right about which
messages to distrust, not to translate them.
"""

from __future__ import annotations

import re

HINDI = {
    "hai",
    "hain",
    "hoon",
    "hun",
    "tha",
    "thi",
    "nahi",
    "nahin",
    "kya",
    "kyu",
    "kyun",
    "mera",
    "meri",
    "mere",
    "mujhe",
    "main",
    "mai",
    "tum",
    "aap",
    "koi",
    "kuch",
    "sab",
    "bas",
    "aur",
    "lekin",
    "par",
    "ki",
    "ka",
    "ke",
    "ko",
    "se",
    "mein",
    "yeh",
    "ye",
    "woh",
    "wo",
    "raha",
    "rahi",
    "rahe",
    "gaya",
    "gayi",
    "karna",
    "karun",
    "karunga",
    "kar",
    "dunga",
    "lunga",
    "hota",
    "hoti",
    "kaise",
    "kab",
    "kahan",
    "yaar",
    "bahut",
    "accha",
    "kaash",
    "zindagi",
    "jeena",
    "marna",
    "khud",
    "apni",
    "apna",
    "liye",
    "abhi",
    "kal",
    "aaj",
    "phir",
    "ghar",
}
TELUGU = {
    "nenu",
    "naa",
    "naaku",
    "nuvvu",
    "meeru",
    "ledu",
    "undi",
    "unnanu",
    "unna",
    "avthunna",
    "chesthanu",
    "chesanu",
    "cheyyali",
    "cheyyaledu",
    "ani",
    "kani",
    "inka",
    "ippudu",
    "repu",
    "ninna",
    "evvaru",
    "evvariki",
    "emi",
    "enduku",
    "ela",
    "chala",
    "konchem",
    "anipistundi",
    "vasthunnayi",
    "chachipovali",
    "chachipotanu",
    "brathakalani",
    "intlo",
    "ayyanu",
    "ayya",
    "ra",
    "amma",
    "nanna",
    "roju",
    "antha",
    "gurinchi",
}
_WORD = re.compile(r"[a-z']+")


def romanised_indic_share(text: str) -> float:
    words = _WORD.findall(text.lower())
    if not words:
        return 0.0
    hits = sum(w in HINDI or w in TELUGU for w in words)
    return hits / len(words)


def is_romanised_indic(text: str, threshold: float = 0.15) -> bool:
    """True when enough of the words are Hindi or Telugu function words.

    Short English words that collide with the lists ("main", "par", "ki",
    "ra") are why the threshold is a share and not a single hit.
    """
    return romanised_indic_share(text) >= threshold
