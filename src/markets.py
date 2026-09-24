"""Превръщане на очакваните голове/корнери във вероятности за пазарите и value."""
from __future__ import annotations

import numpy as np
from scipy.stats import poisson

import config


def score_matrix(lam: float, mu: float, rho: float = 0.0, max_goals: int = config.MAX_GOALS) -> np.ndarray:
    g = np.arange(max_goals + 1)
    m = np.outer(poisson.pmf(g, lam), poisson.pmf(g, mu))
    if rho:
        m[0, 0] *= 1 - lam * mu * rho
        m[0, 1] *= 1 + lam * rho
        m[1, 0] *= 1 + mu * rho
        m[1, 1] *= 1 - rho
    m = np.clip(m, 0, None)
    return m / m.sum()


def goal_markets(m: np.ndarray, lam: float, mu: float) -> dict:
    n = m.shape[0]
    i, j = np.indices(m.shape)
    total = i + j
    out = {
        "xg_home": round(lam, 2), "xg_away": round(mu, 2),
        "p_home": float(np.tril(m, -1).sum()),
        "p_draw": float(np.trace(m)),
        "p_away": float(np.triu(m, 1).sum()),
        "btts_yes": float(m[1:, 1:].sum()),
    }
    out["btts_no"] = 1 - out["btts_yes"]
    for line in config.GOAL_LINES:
        over = float(m[total > line].sum())
        out[f"over_{line}"] = over
        out[f"under_{line}"] = 1 - over
    flat = [(float(m[a, b]), f"{a}-{b}") for a in range(n) for b in range(n)]
    flat.sort(reverse=True)
    out["top_scores"] = [{"score": s, "p": p} for p, s in flat[:6]]
    return out


def corner_markets(lam: float, mu: float) -> dict:
    """Общ брой корнери ~ Poisson(lam + mu) (сума на независими Poisson)."""
    tot = lam + mu
    out = {"xc_home": round(lam, 2), "xc_away": round(mu, 2), "xc_total": round(tot, 2)}
    for line in config.CORNER_LINES:
        over = float(poisson.sf(np.floor(line), tot))
        out[f"c_over_{line}"] = over
        out[f"c_under_{line}"] = 1 - over
    # кой отбор ще има повече корнери
    g = np.arange(31)
    cm = np.outer(poisson.pmf(g, lam), poisson.pmf(g, mu))
    out["c_home_more"] = float(np.tril(cm, -1).sum())
    out["c_away_more"] = float(np.triu(cm, 1).sum())
    return out


def implied(odds: list[float]) -> list[float] | None:
    """Вероятности от коефициентите без маржа на букмейкъра."""
    if any(o is None or not np.isfinite(o) or o <= 1 for o in odds):
        return None
    inv = np.array([1 / o for o in odds])
    return [float(v) for v in inv / inv.sum()]


def blend(p_model: float, p_market: float | None) -> float:
    if p_market is None:
        return p_model
    w = config.VALUE_MODEL_WEIGHT
    return w * p_model + (1 - w) * p_market


def value_bet(market: str, selection: str, p: float, odds: float | None,
              best_odds: float | None = None, p_market: float | None = None) -> dict | None:
    """Връща value залог, ако (смесената) вероятност дава достатъчно предимство."""
    p_model = p
    p = blend(p, p_market)
    price = best_odds if best_odds and np.isfinite(best_odds) else odds
    if price is None or not np.isfinite(price) or price <= 1:
        return None
    edge = p * price - 1
    if edge < config.VALUE_MIN_EDGE or price > config.VALUE_MAX_ODDS or p < config.VALUE_MIN_PROB:
        return None
    kelly = (p * price - 1) / (price - 1)
    return {
        "market": market, "selection": selection, "p": p, "p_model": p_model, "odds": float(price),
        "fair_odds": 1 / p, "edge": edge,
        "stake_pct": round(100 * config.KELLY_FRACTION * kelly, 2),
    }
