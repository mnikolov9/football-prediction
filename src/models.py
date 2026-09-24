"""Статистически модели.

DixonColes  – модел на Dixon & Coles (1997) за голове: Poisson с атака/защита
              на всеки отбор, домакинско предимство, корекция rho за резултати
              0-0, 1-0, 0-1, 1-1 и времево тегло (по-новите мачове тежат повече).
PoissonTeamModel – същата структура без rho; ползва се за корнери.

Силите се регуляризират (L2), което прави модела стабилен за отбори с малко
мачове и еднозначно определя параметрите.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

import config


class PoissonTeamModel:
    use_rho = False

    def __init__(self, xi: float = config.TIME_DECAY_XI, l2: float = config.L2_PENALTY):
        self.xi = xi
        self.l2 = l2
        self.teams: list[str] = []
        self.idx: dict[str, int] = {}
        self.attack = self.defence = None
        self.home_adv = 0.0
        self.rho = 0.0
        self.n_matches: dict[str, int] = {}

    # ------------------------------------------------------------------ #
    def _prepare(self, df, hcol, acol, ref_date):
        df = df.dropna(subset=[hcol, acol])
        self.teams = sorted(set(df["home"]) | set(df["away"]))
        self.idx = {t: i for i, t in enumerate(self.teams)}
        h = df["home"].map(self.idx).to_numpy()
        a = df["away"].map(self.idx).to_numpy()
        x = df[hcol].to_numpy(float)
        y = df[acol].to_numpy(float)
        days = (pd.Timestamp(ref_date) - df["date"]).dt.days.to_numpy(float)
        w = np.exp(-self.xi * np.clip(days, 0, None))
        if "wt" in df:                      # допълнително тегло (напр. приятелски мачове)
            w = w * df["wt"].fillna(1.0).to_numpy(float)
        # 1 = има домакинско предимство, 0 = неутрален терен
        self._hm = 1.0 - df["neutral"].fillna(False).astype(float).to_numpy() if "neutral" in df \
            else np.ones(len(df))
        counts = pd.concat([df["home"], df["away"]]).value_counts()
        self.n_matches = counts.to_dict()
        return h, a, x, y, w

    def _negll(self, p, h, a, x, y, w, n):
        # атаката = общо ниво (base) + отклонение; L2 наказва само отклоненията
        dev, dfn = p[:n], p[n:2 * n]
        att = dev + self._base
        home = p[2 * n]
        rho = p[2 * n + 1] if self.use_rho else 0.0
        e1 = home * self._hm + att[h] - dfn[a]
        e2 = att[a] - dfn[h]
        lam, mu = np.exp(e1), np.exp(e2)

        ll = x * e1 - lam + y * e2 - mu
        g1 = x - lam                     # dLL/d e1
        g2 = y - mu                      # dLL/d e2
        grho = 0.0
        if self.use_rho:
            m00 = (x == 0) & (y == 0)
            m01 = (x == 0) & (y == 1)
            m10 = (x == 1) & (y == 0)
            m11 = (x == 1) & (y == 1)
            tau = np.ones_like(lam)
            tau[m00] = 1 - lam[m00] * mu[m00] * rho
            tau[m01] = 1 + lam[m01] * rho
            tau[m10] = 1 + mu[m10] * rho
            tau[m11] = 1 - rho
            tau = np.clip(tau, 1e-10, None)
            ll = ll + np.log(tau)
            d00 = -lam[m00] * mu[m00] * rho / tau[m00]
            g1[m00] += d00
            g2[m00] += d00
            g1[m01] += lam[m01] * rho / tau[m01]
            g2[m10] += mu[m10] * rho / tau[m10]
            dr = np.zeros_like(lam)
            dr[m00] = -lam[m00] * mu[m00] / tau[m00]
            dr[m01] = lam[m01] / tau[m01]
            dr[m10] = mu[m10] / tau[m10]
            dr[m11] = -1 / tau[m11]
            grho = np.sum(w * dr)

        wg1, wg2 = w * g1, w * g2
        g_att = np.bincount(h, wg1, n) + np.bincount(a, wg2, n)
        g_def = -np.bincount(a, wg1, n) - np.bincount(h, wg2, n)
        g_home = (wg1 * self._hm).sum()

        obj = -np.sum(w * ll) + self.l2 * (dev @ dev + dfn @ dfn)
        grad = np.concatenate([
            -g_att + 2 * self.l2 * dev,
            -g_def + 2 * self.l2 * dfn,
            [-g_home],
            [-grho] if self.use_rho else [],
        ])
        return obj, grad

    def fit(self, df: pd.DataFrame, ref_date=None, hcol="hg", acol="ag"):
        ref_date = ref_date or df["date"].max()
        h, a, x, y, w = self._prepare(df, hcol, acol, ref_date)
        n = len(self.teams)
        base = np.log(max(np.average(np.r_[x, y], weights=np.r_[w, w]), 0.1))
        p0 = np.zeros(2 * n + 1 + int(self.use_rho))
        p0[2 * n] = 0.25
        self._base = base
        bounds = [(-4, 4)] * (2 * n) + [(-1, 2)] + ([(-0.25, 0.25)] if self.use_rho else [])
        res = minimize(self._negll, p0, args=(h, a, x, y, w, n), jac=True,
                       method="L-BFGS-B", bounds=bounds, options={"maxiter": 2000})
        p = res.x
        self.attack = p[:n] + base
        self.defence = p[n:2 * n]
        self.home_adv = p[2 * n]
        self.rho = p[2 * n + 1] if self.use_rho else 0.0
        self.converged = res.success
        return self

    # ------------------------------------------------------------------ #
    def knows(self, team: str) -> bool:
        return team in self.idx

    def rates(self, home: str, away: str, neutral: bool = False) -> tuple[float, float]:
        """Очаквани стойности (домакин, гост)."""
        ah = self.attack[self.idx[home]] if home in self.idx else self._unknown_attack()
        dh = self.defence[self.idx[home]] if home in self.idx else self._unknown_defence()
        aa = self.attack[self.idx[away]] if away in self.idx else self._unknown_attack()
        da = self.defence[self.idx[away]] if away in self.idx else self._unknown_defence()
        hadv = 0.0 if neutral else self.home_adv
        return float(np.exp(hadv + ah - da)), float(np.exp(aa - dh))

    # непознат отбор = по-слаб от средното (обикновено новак от по-долна лига)
    def _unknown_attack(self):
        return float(np.percentile(self.attack, 20))

    def _unknown_defence(self):
        return float(np.percentile(self.defence, 20))

    def ratings(self) -> pd.DataFrame:
        return pd.DataFrame({
            "team": self.teams, "attack": self.attack, "defence": self.defence,
            "matches": [self.n_matches.get(t, 0) for t in self.teams],
        }).assign(strength=lambda d: d.attack + d.defence).sort_values("strength", ascending=False)


class DixonColes(PoissonTeamModel):
    use_rho = True
