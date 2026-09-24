"""Gradient boosting (LightGBM) слой върху Dixon–Coles.

Идея: Dixon–Coles дава очакваните голове lam (домакин) и mu (гост). Два
LightGBM модела с Poisson цел научават КОРЕКЦИЯ (множител) на lam и mu според
формата, xG, „късмета“ (голове над/под xG) и почивката. Коригираните lam/mu
минават през същата матрица с точни резултати, така всички пазари (1X2,
над/под, ГГ, точен резултат) остават съгласувани.

Обучението става само върху прогнози „извън извадката“: Dixon–Coles се
обучава само с мачове преди всеки период (walk-forward). Последната част от
периода се пази за проверка – слоят се включва за дадено първенство само ако
там има по-нисък log loss 1X2 от чистия Dixon–Coles.

Poisson с изместване: максимизирането на y*log(lam*m) - lam*m е същото като
Poisson регресия на y/lam с тегло lam. Така моделът учи директно множителя m.
"""
from __future__ import annotations

import pickle

import numpy as np
import pandas as pd

import config
from src import features, markets

MULT_CLIP = (0.6, 1.6)          # предпазител срещу крайни корекции
MIN_ROWS = 400                  # минимум мачове за обучение + проверка
# (брой дървета, минимум мачове в листо) – от по-консервативно към по-гъвкаво.
# Изборът става върху вътрешна проверка в периода за обучение.
GRID = [(20, 200), (50, 150), (100, 100)]

_SLUGS = {c: next(iter(divs)).lower() for c, divs in config.COUNTRIES.items()}
_SLUGS.update({config.CL_NAME: "cl", config.NL_NAME: "unl"})
_cache: dict[str, object] = {}


def _estimator(n_trees: int, min_leaf: int):
    try:
        import lightgbm as lgb
        return lgb.LGBMRegressor(objective="poisson", n_estimators=n_trees, learning_rate=0.03,
                                 num_leaves=8, min_child_samples=min_leaf, subsample=0.8, subsample_freq=1,
                                 colsample_bytree=0.8, reg_lambda=5.0, verbose=-1)
    except ImportError:          # резервен вариант, ако lightgbm не е инсталиран
        from sklearn.ensemble import HistGradientBoostingRegressor
        return HistGradientBoostingRegressor(loss="poisson", max_iter=n_trees, learning_rate=0.03,
                                             max_leaf_nodes=8, min_samples_leaf=min_leaf,
                                             l2_regularization=5.0)


class GoalBoost:
    def __init__(self, n_trees: int = 50, min_leaf: int = 150):
        self.cfg = (n_trees, min_leaf)

    def fit(self, X: pd.DataFrame, hg, ag, lam, mu):
        lam, mu = np.asarray(lam, float), np.asarray(mu, float)
        self.mh = _estimator(*self.cfg).fit(X, np.asarray(hg, float) / lam, sample_weight=lam)
        self.ma = _estimator(*self.cfg).fit(X, np.asarray(ag, float) / mu, sample_weight=mu)
        return self

    def adjust(self, X: pd.DataFrame, lam, mu) -> tuple[np.ndarray, np.ndarray]:
        fh = np.clip(self.mh.predict(X), *MULT_CLIP)
        fa = np.clip(self.ma.predict(X), *MULT_CLIP)
        return np.asarray(lam, float) * fh, np.asarray(mu, float) * fa


def _logloss(df: pd.DataFrame, lam, mu) -> tuple[float, float]:
    """Log loss за 1X2 и за над/под 2.5."""
    l1, l2 = [], []
    for (hg, ag, rho), l, m in zip(df[["hg", "ag", "rho"]].itertuples(index=False), lam, mu):
        g = markets.goal_markets(markets.score_matrix(l, m, rho), l, m)
        p = g["p_home"] if hg > ag else g["p_draw"] if hg == ag else g["p_away"]
        po = g["over_2.5"] if hg + ag > 2.5 else g["under_2.5"]
        l1.append(-np.log(max(p, 1e-9)))
        l2.append(-np.log(max(po, 1e-9)))
    return float(np.mean(l1)), float(np.mean(l2))


def _split(dates: pd.Series, share: float) -> tuple[pd.Series, pd.Series]:
    """Разделяне по време: последните share от мачовете са за проверка."""
    cut = dates.iloc[int(len(dates) * (1 - share))]
    return dates < cut, dates >= cut


def _fit_eval(oos, X, tr, va, cfg) -> tuple[GoalBoost, np.ndarray, np.ndarray]:
    b = GoalBoost(*cfg).fit(X[tr], oos.hg[tr], oos.ag[tr], oos.lam[tr], oos.mu[tr])
    lam2, mu2 = b.adjust(X[va], oos.lam[va], oos.mu[va])
    return b, lam2, mu2


def train(oos: pd.DataFrame, X: pd.DataFrame) -> tuple[GoalBoost | None, dict]:
    """oos – walk-forward прогнози на Dixon–Coles (с lam, mu, rho, hg, ag, date).

    Трите части са подредени по време: [обучение | вътрешна проверка] -> избор на
    настройките; [обучение] -> [проверка] -> решение вкл./изкл. спрямо Dixon–Coles."""
    oos = oos.sort_values("date", kind="stable")
    X = X.loc[oos.index]
    if len(oos) < MIN_ROWS:
        return None, {"enabled": False, "reason": f"малко мачове ({len(oos)})", "n": int(len(oos))}
    tr, va = _split(oos["date"], config.BOOST_VALID_SHARE)
    # избор на настройките само в периода за обучение
    sub = oos[tr]
    itr, iva = _split(sub["date"], 0.3)
    itr, iva = itr.reindex(oos.index, fill_value=False), iva.reindex(oos.index, fill_value=False)
    scores = {cfg: _logloss(oos[iva], *_fit_eval(oos, X, itr, iva, cfg)[1:])[0] for cfg in GRID}
    cfg = min(scores, key=scores.get)
    _, lam2, mu2 = _fit_eval(oos, X, tr, va, cfg)
    ll_dc, ou_dc = _logloss(oos[va], oos.lam[va], oos.mu[va])
    ll_gb, ou_gb = _logloss(oos[va], lam2, mu2)
    enabled = ll_dc - ll_gb >= config.BOOST_MIN_GAIN
    info = {"enabled": bool(enabled), "n_train": int(tr.sum()), "n_valid": int(va.sum()),
            "logloss_dc": ll_dc, "logloss_boost": ll_gb, "ou_logloss_dc": ou_dc, "ou_logloss_boost": ou_gb,
            "trees": cfg[0], "min_leaf": cfg[1],
            "xg_share": float(X[["h_xgf10", "a_xgf10"]].notna().all(1).mean())}
    if not enabled:
        info["reason"] = "не подобрява Dixon–Coles"
        return None, info
    # окончателният модел се учи върху целия период
    return GoalBoost(*cfg).fit(X, oos.hg, oos.ag, oos.lam, oos.mu), info


def _path(name: str):
    return config.BOOST_DIR / f"{_SLUGS.get(name, name)}.pkl"


def save(name: str, model: GoalBoost | None):
    p = _path(name)
    if model is None:
        p.unlink(missing_ok=True)       # изключен слой -> файлът се маха
        return
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(pickle.dumps(model))


def load(name: str) -> GoalBoost | None:
    if not config.BOOST_ENABLED:
        return None
    if name not in _cache:
        p = _path(name)
        try:
            _cache[name] = pickle.loads(p.read_bytes()) if p.exists() else None
        except Exception as e:          # noqa: BLE001 – повреден/несъвместим файл
            print(f"  ! Boost модел {p.name}: {e}")
            _cache[name] = None
    return _cache[name]


def tune_group(name: str, hist: pd.DataFrame, xi: float, l2: float, step: int = 14) -> dict:
    """Walk-forward прогнози -> характеристики -> обучение и проверка -> запис."""
    from src.backtest import walk_forward         # тук, за да няма кръгов импорт
    oos = walk_forward(hist, xi=xi, l2=l2, test_days=config.BOOST_TRAIN_DAYS, step=step)
    if oos.empty:
        save(name, None)
        return {"enabled": False, "reason": "няма данни"}
    X = features.build(hist, oos)
    model, info = train(oos, X)
    save(name, model)
    _cache.pop(name, None)
    return info


def apply(name: str, played: pd.DataFrame, fx: pd.DataFrame, dc) -> pd.DataFrame:
    """Добавя lam_adj/mu_adj към предстоящите мачове, ако слоят е включен."""
    model = load(name)
    if model is None or fx.empty:
        return fx
    fx = fx.copy()
    neutral = fx["neutral"].fillna(False).astype(bool) if "neutral" in fx else pd.Series(False, index=fx.index)
    rates = [dc.rates(h, a, neutral=n) for h, a, n in zip(fx["home"], fx["away"], neutral)]
    fx["lam"], fx["mu"] = [r[0] for r in rates], [r[1] for r in rates]
    X = features.build(played, fx)
    fx["lam_adj"], fx["mu_adj"] = model.adjust(X, fx["lam"], fx["mu"])
    return fx
