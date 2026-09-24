"""Команден ред.

  python run.py all           # сваля данни, прави прогнози, генерира сайта
  python run.py backtest      # проверка на точността на модела върху минали мачове
  python run.py all --offline # без интернет, с вече свалените (или синтетични) данни
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import time

import config
from src import backtest, predict, site, tracking


def cmd_all(offline: bool):
    t0 = time.time()
    today = dt.date.today()
    result, hist = predict.run(offline=offline, today=today)
    tracking.log_predictions(result, today.isoformat())
    result["track_record"] = tracking.evaluate(hist)
    bt_path = config.DATA_DIR / "backtest.json"
    result["backtest"] = json.loads(bt_path.read_text("utf-8")) if bt_path.exists() else {}
    site.build(result)
    print(f"Готово: {len(result['matches'])} мача, {len(result['value_bets'])} value залога "
          f"({time.time() - t0:.0f} сек). Сайтът е в {config.SITE_DIR}")


def cmd_backtest(offline: bool):
    res = backtest.run(offline=offline)
    res["_generated"] = dt.date.today().isoformat()
    config.DATA_DIR.mkdir(exist_ok=True)
    (config.DATA_DIR / "backtest.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), "utf-8")
    print(f"Записано в {config.DATA_DIR / 'backtest.json'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["all", "backtest"])
    ap.add_argument("--offline", action="store_true")
    a = ap.parse_args()
    {"all": cmd_all, "backtest": cmd_backtest}[a.command](a.offline)
