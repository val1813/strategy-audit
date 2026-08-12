"""端到端验证：在真实 A 股面板上造两版同一个策略，让审计器区分。

面板   ashare_historical_daily_2015_2026.parquet（800 只 / 2015-2026 / 2787 交易日）
策略   月频、20 日反转、选 30 只

两版的【投资逻辑完全相同】，只有实现细节不同：

    careless   信号用调仓日收盘算，权重按【调仓日】市值加权，
               股票池取「样本末仍在册」的名单
    careful    信号滞后一日，权重按【上一调仓日】市值加权，
               股票池按调仓日当时可见的标的

careless 那三处都是我在实测里踩过的真实漏口（见记忆
vw-weight-and-universe-filter-lookahead），不是人为夸张的构造。

★ 本脚本只负责【生成权重】，不做任何审计判断。
审计结论全部来自 strategy_audit，这样才算端到端验证。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PANEL = (r"D:/huang/Pseudo-Idiosyncratic Risk - Retail Dominance in China"
         r"/revision_outputs/ashare_historical_daily_2015_2026.parquet")

LOOKBACK = 20      # 反转信号回看交易日
N_HOLD = 30        # 持仓数
START = "2016-01-01"   # 留出回看窗口


def load_panel() -> pd.DataFrame:
    d = pd.read_parquet(PANEL)
    d["date"] = pd.to_datetime(d["date"])
    return d.sort_values(["code", "date"]).reset_index(drop=True)


def month_end_dates(dates: pd.DatetimeIndex) -> list:
    s = pd.Series(dates.unique()).sort_values()
    df = pd.DataFrame({"d": s})
    df["ym"] = df["d"].dt.to_period("M")
    return list(df.groupby("ym")["d"].max())


def build(d: pd.DataFrame, careless: bool) -> pd.DataFrame:
    """生成月频权重。careless=True 时植入三处真实漏口。"""
    px = d.pivot(index="date", columns="code", values="close").sort_index()
    cap = d.pivot(index="date", columns="code", values="cap").sort_index()

    reb = [t for t in month_end_dates(px.index) if t >= pd.Timestamp(START)]
    pos = {t: i for i, t in enumerate(px.index)}

    # ★ 漏口三：股票池。careless 用「样本末仍在册」的名单 —— 用了未来信息。
    if careless:
        universe = set(px.columns[px.iloc[-1].notna()])
    else:
        universe = None        # 每期按当时可见的标的决定

    rows = []
    prev_t = None
    for t in reb:
        i = pos[t]
        if i < LOOKBACK + 1:
            prev_t = t
            continue

        # ★ 漏口一：信号时点。
        # careless 用调仓日收盘价算信号，同一天就按它下单（收盘价拿到时已收市）。
        # careful 用前一交易日收盘，t 日开盘可下单。
        sig_end = i if careless else i - 1
        sig = px.iloc[sig_end] / px.iloc[sig_end - LOOKBACK] - 1.0
        sig = -sig                                   # 反转：跌得多的买

        avail = px.iloc[i].notna() & sig.notna()
        if universe is not None:
            avail &= px.columns.isin(universe)
        pool = px.columns[avail]
        if len(pool) < N_HOLD:
            prev_t = t
            continue

        pick = sig[pool].nlargest(N_HOLD).index

        # ★ 漏口二：权重口径。
        # careless 按【调仓日】市值加权 —— 当日 cap 含当日收益。
        # careful 按【上一调仓日】市值加权，那是下单时真能看到的。
        cap_t = t if (careless or prev_t is None) else prev_t
        w = cap.loc[cap_t, pick].astype(float)
        w = w.where(np.isfinite(w) & (w > 0))
        if w.isna().any() or float(w.sum()) <= 0:
            w = pd.Series(1.0, index=pick)
        w = w / w.sum()

        for c, ww in w.items():
            rows.append((t, c, float(ww)))
        prev_t = t

    return pd.DataFrame(rows, columns=["date", "code", "weight"])


def main() -> int:
    from strategy_audit import audit_strategy, core

    d = load_panel()
    prices = d[["date", "code", "close"]].copy()
    print("面板 {:,} 行 / {} 只 / {} ~ {}".format(
        len(d), d["code"].nunique(), d["date"].min().date(), d["date"].max().date()))

    for careless in (True, False):
        tag = "careless（植入三处真实漏口）" if careless else "careful（三处都修好）"
        w = build(d, careless=careless)

        # 组合自身的表现（审计器之外，供对照）
        wm = w.pivot(index="date", columns="code", values="weight").fillna(0.0)
        wm = wm.reindex(sorted(wm.columns), axis=1)
        pm = core.price_matrix(prices)
        pr = core.period_returns(wm, pm)
        ppy = core.periods_per_year(wm.index)
        ann = core.annualize(pr["ret"], ppy)

        print("\n" + "=" * 72)
        print("  " + tag)
        print("=" * 72)
        print("  {} 期 / 毛年化 {:.2%} / Sharpe {:.2f}".format(
            len(wm.index), ann["ann_ret"], ann["sharpe"]))

        rep = audit_strategy(w, prices, name=f"真实A股 · {tag}")
        print(rep.text())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
