"""Съпоставяне на имената на отборите между football-data.org и football-data.co.uk."""
from __future__ import annotations

import difflib
import re
import unicodedata

# думи без значение за съпоставянето
STOP = {"fc", "afc", "cf", "sc", "ac", "as", "ssc", "ss", "us", "sv", "vfb", "vfl", "tsg", "fsv",
        "rc", "ogc", "cd", "ud", "rcd", "sd", "ca", "club", "de", "calcio", "bc", "1", "sco", "aj",
        "ec", "fk", "hsc", "stade", "olympique", "real", "football", "and", "hove", "albion"}

# трудни случаи: нормализирано име от football-data.org -> нормализирано име от .co.uk
ALIASES = {
    "psg": "paris sg", "paris saint germain": "paris sg",
    "barca": "barcelona", "atleti": "ath madrid", "atletico madrid": "ath madrid",
    "athletic": "ath bilbao", "athletic bilbao": "ath bilbao", "sociedad": "sociedad",
    "nottingham": "nottm forest", "nottingham forest": "nottm forest",
    "wolverhampton": "wolves", "wolverhampton wanderers": "wolves",
    "manchester united": "man united", "manchester city": "man city",
    "sheffield wednesday": "sheffield weds", "queens park rangers": "qpr",
    "west bromwich": "west brom", "bayern": "bayern munich", "munchen": "bayern munich",
    "eintracht frankfurt": "ein frankfurt", "frankfurt": "ein frankfurt",
    "borussia monchengladbach": "mgladbach", "monchengladbach": "mgladbach",
    "koln": "fc koln", "koeln": "fc koln", "hsv": "hamburg", "hamburger": "hamburg",
    "inter milan": "inter", "internazionale": "inter", "milan": "milan",
    "hellas verona": "verona", "rayo vallecano": "vallecano", "espanyol": "espanol",
    "celta vigo": "celta", "saint etienne": "st etienne", "st pauli": "st pauli",
}


def norm(name: str) -> str:
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"[^a-z0-9 ]", " ", s.replace("'", ""))
    words = [w for w in s.split() if w not in STOP and not re.fullmatch(r"\d{2,4}", w)]
    s = " ".join(words)
    return ALIASES.get(s, s)


def match(variants: list[str], candidates: list[str], cutoff: float = 0.72) -> str | None:
    """Връща името от candidates, което най-добре отговаря на някой от вариантите."""
    cand = {norm(c): c for c in candidates}
    best, best_score = None, 0.0
    for v in filter(None, variants):
        n = norm(v)
        if n in cand:
            return cand[n]
        for cn, orig in cand.items():
            score = difflib.SequenceMatcher(None, n, cn).ratio()
            # "tottenham hotspur" съдържа "tottenham"
            if n and cn and (n in cn or cn in n) and min(len(n), len(cn)) >= 4:
                score = max(score, 0.9)
            if score > best_score:
                best, best_score = orig, score
    return best if best_score >= cutoff else None
