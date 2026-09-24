"""Генерира статичния сайт (site/index.html + site/data.json)."""
from __future__ import annotations

import json
import math

from jinja2 import Environment, FileSystemLoader

import config


def _clean(o):
    """NaN/inf -> None, за да е валиден JSON."""
    if isinstance(o, float):
        return None if not math.isfinite(o) else round(o, 4)
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if hasattr(o, "item"):          # numpy скалари
        return _clean(o.item())
    return o


def build(result: dict):
    data = _clean(result)
    config.SITE_DIR.mkdir(parents=True, exist_ok=True)
    (config.SITE_DIR / "data.json").write_text(json.dumps(data, ensure_ascii=False), "utf-8")
    env = Environment(loader=FileSystemLoader(config.TEMPLATES_DIR), autoescape=True)
    html = env.get_template("index.html").render(
        data_json=json.dumps(data, ensure_ascii=False).replace("</", "<\\/"),
        generated=data["generated"],
    )
    (config.SITE_DIR / "index.html").write_text(html, "utf-8")
    (config.SITE_DIR / ".nojekyll").write_text("")
