"""Характеристики за boosting слоя: форма, голове, xG, „късмет“ и почивка.

Всички стойности за даден мач се смятат само от мачове, изиграни ПРЕДИ
датата му, така че няма изтичане на информация от бъдещето.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config

# ниво на дивизията (0 = първа) – моделът може да се държи различно по нива
TIER = {d: i for divs in config.COUNTRIES.values() for i, d in enumerate(divs)}
TIER.update({config.CL_CODE: 0, config.NL_CODE: 0, "INT": 1})

_STATS = ["pts5", "pts10", "gf10", "ga10", "xgf10", "xga10", "luck10", "n"]
FEATURES = (["dc_lh", "dc_la", "tier", "neutral", "rest_diff"]
            + [f"{s}_{c}" for s in ("h", "a") for c in _STATS + ["rest"]])


def _long(played: pd.DataFrame) -> pd.DataFrame:
    """Един ред на отбор и мач, с плъзгащи се средни СЛЕД този мач."""
    hxg = played["hxg"] if "hxg" in played else np.nan
    axg = played["axg"] if "axg" in played else np.nan
    h = pd.DataFrame({"team": played["home"], "date": played["date"], "gf": played["hg"],
                      "ga": played["ag"], "xgf": hxg, "xga": axg})
    a = pd.DataFrame({"team": played["away"], "date": played["date"], "gf": played["ag"],
                      "ga": played["hg"], "xgf": axg, "xga": hxg})
    L = pd.concat([h, a], ignore_index=True)
    L[["gf", "ga", "xgf", "xga"]] = L[["gf", "ga", "xgf", "xga"]].astype(float)
    L["pts"] = np.select([L.gf > L.ga, L.gf == L.ga], [3.0, 1.0], 0.0)
    L["luck"] = L.gf - L.xgf                       # вкарани над/под xG
    L = L.sort_values(["team", "date"], kind="stable").reset_index(drop=True)
    g = L.groupby("team", sort=False)

    def roll(col, w, mp=1):
        return g[col].transform(lambda s: s.rolling(w, min_periods=mp).mean())

    L["pts5"], L["pts10"] = roll("pts", 5), roll("pts", 10)
    L["gf10"], L["ga10"] = roll("gf", 10), roll("ga", 10)
    L["xgf10"], L["xga10"] = roll("xgf", 10, 3), roll("xga", 10, 3)
    L["luck10"] = roll("luck", 10, 3)
    L["n"] = g.cumcount() + 1.0
    return L[["team", "date"] + _STATS]


def _asof(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    """За всеки ред от left – последното състояние на отбора строго преди датата."""
    left = left.sort_values("date", kind="stable")
    right = right.sort_values("date", kind="stable")
    return pd.merge_asof(left, right, on="date", by="team", allow_exact_matches=False)


def build(played: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
    """played – изиграни мачове (date, home, away, hg, ag[, hxg, axg]).
    targets – мачове за прогноза (date, div, home, away, lam, mu[, neutral]);
    могат да са и изиграни (при обучение). Връща FEATURES, подредени като targets."""
    t = targets.copy()
    t["_qid"] = np.arange(len(t))
    # мачове без дата (Лига на нациите) – като ден след последния известен
    fill = pd.to_datetime(pd.concat([played["date"], t["date"]])).max()
    fill = (pd.Timestamp.today().normalize() if pd.isna(fill) else fill) + pd.Timedelta(days=1)
    t["date"] = pd.to_datetime(t["date"]).fillna(fill).astype("datetime64[ns]")
    played = played.dropna(subset=["hg", "ag"]).copy()
    played["date"] = pd.to_datetime(played["date"]).astype("datetime64[ns]")

    L = _long(played)
    L["date"] = L["date"].astype("datetime64[ns]")
    # график за почивката: изиграните мачове + предстоящите от targets
    sched = pd.concat([pd.DataFrame({"team": s, "date": d}) for s, d in
                       [(played.home, played.date), (played.away, played.date),
                        (t.home, t.date), (t.away, t.date)]], ignore_index=True)
    sched = sched.drop_duplicates().assign(last=lambda d: d["date"])

    out = pd.DataFrame(index=t["_qid"])
    for side, col in (("h", "home"), ("a", "away")):
        q = t[["_qid", "date", col]].rename(columns={col: "team"})
        st = _asof(q, L).set_index("_qid")
        for c in _STATS:
            out[f"{side}_{c}"] = st[c]
        out[f"{side}_n"] = out[f"{side}_n"].fillna(0.0)
        rs = _asof(q, sched[["team", "date", "last"]]).set_index("_qid")
        out[f"{side}_rest"] = ((rs["date"] - rs["last"]).dt.days.clip(0, 30)).astype(float)
    out["rest_diff"] = out["h_rest"] - out["a_rest"]
    out["dc_lh"] = np.log(t["lam"].to_numpy(float))
    out["dc_la"] = np.log(t["mu"].to_numpy(float))
    out["tier"] = t["div"].map(TIER).fillna(0).to_numpy(float)
    nt = t["neutral"] if "neutral" in t else pd.Series(False, index=t.index)
    out["neutral"] = nt.fillna(False).astype(float).to_numpy()
    out = out.sort_index()
    out.index = targets.index
    return out[FEATURES]
