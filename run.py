"""Команден ред.

  python run.py all           # сваля данни, прави прогнози, генерира сайта
  python run.py tune          # настройка на модела и value правилата (~10 мин)
  python run.py backtest      # проверка на точността на модела върху минали мачове
  python run.py all --offline # без интернет, с вече свалените (или синтетични) данни
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import time

import config
from src import backtest, bank, motd, params, predict, site, tracking, tuning


def cmd_all(offline: bool):
    t0 = time.time()
    today = dt.date.today()
    result, hist = predict.run(offline=offline, today=today)
    tracking.log_predictions(result, today.isoformat())
    result["track_record"] = tracking.evaluate(hist)
    bt_path = config.DATA_DIR / "backtest.json"
    result["backtest"] = json.loads(bt_path.read_text("utf-8")) if bt_path.exists() else {}
    # таб „Банка“: по един залог на ден (най-вероятният изход)
    picks = bank.daily_picks(result["matches"])
    bank.log_daily(picks, today.isoformat())
    result["bank"] = {
        "start": config.BANK_START, "daily_pct": config.BANK_DAILY_PCT,
        "live": bank.history(hist),
        "backtest": bank.best_per_day([b for r in result["backtest"].values() if isinstance(r, dict)
                                       for b in r.pop("daily_log", [])]),
    }
    # таб „Мач на деня“: един мач с коефициент над 2.00 + анализ
    motd.log_daily(motd.daily_picks(result["matches"]), today.isoformat())
    result["motd"] = motd.build(result, hist, today.isoformat())
    # таб „Минали мачове“: прогноза срещу резултат
    result["past"] = tracking.past_results(hist, days=config.PAST_DAYS, days_nations=config.PAST_DAYS_NATIONS)
    result["past_days"] = config.PAST_DAYS
    result["past_days_nations"] = config.PAST_DAYS_NATIONS
    result["tuning"] = params.load()
    site.build(result)
    print(f"Готово: {len(result['matches'])} мача, {len(result['value_bets'])} value залога "
          f"({time.time() - t0:.0f} сек). Сайтът е в {config.SITE_DIR}")


def cmd_backtest(offline: bool):
    res = backtest.run(offline=offline)
    res["_generated"] = dt.date.today().isoformat()
    config.DATA_DIR.mkdir(exist_ok=True)
    (config.DATA_DIR / "backtest.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), "utf-8")
    print(f"Записано в {config.DATA_DIR / 'backtest.json'}")


def cmd_tune(offline: bool):
    tuning.run(offline=offline)
    print(f"Записано в {params.TUNED}. Бектестът се обновява с новите параметри...")
    cmd_backtest(offline)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["all", "backtest", "tune"])
    ap.add_argument("--offline", action="store_true")
    a = ap.parse_args()
    {"all": cmd_all, "backtest": cmd_backtest, "tune": cmd_tune}[a.command](a.offline)
