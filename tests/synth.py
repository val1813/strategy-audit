"""合成面板工厂。

★ 固定种子 —— 测试必须逐次可复现，不用时钟或漂移的随机源。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_prices(n_codes=40, start="2021-01-04", end="2023-12-29",
                seed=11, mu=0.0003, sigma=0.018, n_dead=0,
                death_drawdown=0.0):
    """日频价格面板（矩阵形式）。n_dead>0 时让前 n_dead 只在中途消失。

    death_drawdown  退市前 20 个交易日的累计跌幅（0.8 = 跌 80%）。

    ★ 这个参数必须存在，否则合成面板测不出真实机制。
    真实退市股在消失【之前】会崩盘（ST → 面值退市，常见 −50%~−90%）。
    让标的在中性价位凭空消失，就没有「亏损」可供无损移除去扔掉 ——
    于是 drop 与 hold_last 的高低随存活股当期涨跌而变，测不出定向结论。
    实测：death_drawdown=0 时 6 个面板里 3 个出现 drop < hold_last。
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, end)
    codes = [f"{i:06d}.SZ" for i in range(1, n_codes + 1)]
    px = pd.DataFrame(
        {c: 100 * np.exp(np.cumsum(rng.normal(mu, sigma, len(dates))))
         for c in codes}, index=dates)
    for i in range(n_dead):
        # 在 40%~80% 的位置退市，之后价格缺失
        cut = int(len(dates) * (0.4 + 0.4 * (i + 1) / max(n_dead, 1)))
        if death_drawdown > 0:
            # 退市前 20 日逐步崩盘
            k = min(20, cut)
            decay = (1.0 - death_drawdown) ** (np.arange(1, k + 1) / k)
            px.iloc[cut - k:cut, i] = px.iloc[cut - k:cut, i].values * decay
        px.iloc[cut:, i] = np.nan
    return px


def month_ends(px: pd.DataFrame) -> list:
    """价格面板里每个月的最后一个交易日。"""
    out = []
    for d in px.resample("ME").last().index:
        i = px.index.get_indexer([d], method="ffill")[0]
        if i >= 0:
            out.append(px.index[i])
    return sorted(set(out))


def to_long(m: pd.DataFrame, value: str) -> pd.DataFrame:
    """矩阵 → 长表（权重去零，价格去缺失）。"""
    s = m.stack()
    s = s[s != 0.0] if value == "weight" else s.dropna()
    out = s.rename(value).reset_index()
    out.columns = ["date", "code", value]
    return out


def _pivot(rows):
    m = (pd.DataFrame(rows, columns=["date", "code", "weight"])
         .pivot(index="date", columns="code", values="weight")
         .fillna(0.0).sort_index())
    return m.reindex(sorted(m.columns), axis=1)


def equal_weight(px, reb, k=10, seed=7):
    """等权随机持仓（干净基准）。"""
    rng = np.random.default_rng(seed)
    rows = []
    for t in reb:
        pool = [c for c in px.columns if np.isfinite(px.loc[t, c])]
        if len(pool) < k:
            continue
        for c in rng.choice(pool, k, replace=False):
            rows.append((t, c, 1.0 / k))
    return _pivot(rows)


def tilted_weight(px, reb, k=10, seed=7):
    """权重有截面变化但【不】含前视：按上期收益（已实现）递减加权。"""
    rng = np.random.default_rng(seed)
    rows = []
    prev = None
    for t in reb:
        pool = [c for c in px.columns if np.isfinite(px.loc[t, c])]
        if len(pool) < k:
            prev = t
            continue
        pick = list(rng.choice(pool, k, replace=False))
        if prev is None:
            w = np.full(k, 1.0 / k)
        else:
            # 只用【过去】信息排序：上一调仓日到本日的已实现收益
            past = (px.loc[t, pick] / px.loc[prev, pick] - 1.0).fillna(0.0)
            order = np.argsort(-past.values)
            w = np.zeros(k)
            ramp = np.linspace(0.19, 0.01, k)
            w[order] = ramp / ramp.sum()
        for c, ww in zip(pick, w):
            rows.append((t, c, float(ww)))
        prev = t
    return _pivot(rows)


def lookahead_weight(px, reb, k=10):
    """植入前视：选下期赢家，且按下期收益递减加权。"""
    rows = []
    for t0, t1 in zip(reb[:-1], reb[1:]):
        fwd = (px.loc[t1] / px.loc[t0] - 1.0).dropna().sort_values(ascending=False)
        win = list(fwd.index[:k])
        if len(win) < k:
            continue
        ramp = np.linspace(0.19, 0.01, k)
        ramp = ramp / ramp.sum()
        for c, ww in zip(win, ramp):
            rows.append((t0, c, float(ww)))
    return _pivot(rows)
