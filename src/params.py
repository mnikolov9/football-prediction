"""Настроени параметри (data/tuned.json, създава се от `python run.py tune`)."""
from __future__ import annotations

import json

import config

TUNED = config.DATA_DIR / "tuned.json"


def load() -> dict:
    try:
        return json.loads(TUNED.read_text("utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def for_group(name: str) -> dict:
    d = load().get(name, {})
    return {"xi": d.get("xi", config.TIME_DECAY_XI),
            "l2": d.get("l2", config.L2_PENALTY),
            "value": d.get("value", {})}
