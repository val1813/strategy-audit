"""
Multi-factor long-only stock selection backtest on A-shares (800-stock panel, 2015-2026).

Strategy sketch
---------------
Monthly rebalance (last trading day of each calendar month).
Four cross-sectional factors, z-scored and equally weighted into a composite:
    1. 20-day short-term reversal      : -cumulative return over the trailing 20 sessions
    2. Size                            : -log(market cap)          (small-cap tilt)
    3. Turnover                        : -mean(turn) over 20 sessions (low-attention tilt)
    4. Volatility                      : -std(ret) over 60 sessions  (low-vol tilt)
Top 50 composite scores, equal weighted, held for one month.
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd

# 原始行情不随仓库分发。运行本复现脚本前，显式指定自己的日频面板路径：
#   ASHARE_PANEL=/path/to/daily.parquet python _blind/strategy.py
PARQUET = Path(os.environ.get("ASHARE_PANEL", "path/to/daily.parquet"))
OUTDIR = Path(os.environ.get("STRATEGY_AUDIT_OUTDIR",
                             str(Path(__file__).resolve().parent)))

# ---- strategy parameters -------------------------------------------------
REV_WIN = 20          # reversal lookback (sessions)
TURN_WIN = 20         # turnover lookback (sessions)
VOL_WIN = 60          # volatility lookback (sessions)
MIN_HIST = 120        # min sessions of history required to be eligible
N_HOLD = 50           # number of names held
MIN_AMOUNT = 5e7      # min 20d average daily turnover value (CNY) for liquidity screen
WINSOR = 3.0          # z-score clip


def zscore(s):
    s = s.astype(float)
    z = (s - s.mean()) / s.std(ddof=0)
    return z.clip(-WINSOR, WINSOR)


def main():
    if not PARQUET.is_file():
        raise SystemExit(
            "找不到 A 股日频面板。请设置 ASHARE_PANEL，"
            "例如 ASHARE_PANEL=/path/to/daily.parquet。")
    os.makedirs(OUTDIR, exist_ok=True)

    cols = ["date", "code", "close", "amount", "turn", "ret", "cap"]
    df = pd.read_parquet(PARQUET, columns=cols)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "code"]).reset_index(drop=True)

    # ---- wide panels ----------------------------------------------------
    ret = df.pivot(index="date", columns="code", values="ret")
    turn = df.pivot(index="date", columns="code", values="turn")
    amount = df.pivot(index="date", columns="code", values="amount")
    cap = df.pivot(index="date", columns="code", values="cap")
    close = df.pivot(index="date", columns="code", values="close")

    dates = ret.index
    n_dates = len(dates)

    # ---- rolling factor inputs (all computed on data up to and including t)
    logret = np.log1p(ret)
    rev = logret.rolling(REV_WIN).sum()                 # trailing 20d log return
    turn_ma = turn.rolling(TURN_WIN).mean()
    vol = ret.rolling(VOL_WIN).std()
    amt_ma = amount.rolling(TURN_WIN).mean()
    hist = ret.notna().rolling(MIN_HIST).sum()          # observed sessions in trailing window

    # ---- rebalance dates: last trading day of each month -----------------
    month_id = dates.to_period("M")
    last_of_month = pd.Series(dates, index=dates).groupby(month_id).max().values
    rebal_dates = [d for d in pd.DatetimeIndex(last_of_month) if d >= dates[VOL_WIN + 5]]
    # keep only rebalances that have at least one following trading day
    rebal_dates = [d for d in rebal_dates if d < dates[-1]]

    pos_of = {d: i for i, d in enumerate(dates)}

    weight_rows = []
    prev_w = pd.Series(dtype=float)
    turnovers = []

    for d in rebal_dates:
        elig = (
            ret.loc[d].notna()
            & rev.loc[d].notna()
            & turn_ma.loc[d].notna()
            & vol.loc[d].notna()
            & (hist.loc[d] >= MIN_HIST * 0.9)
            & (amt_ma.loc[d] >= MIN_AMOUNT)
            & (vol.loc[d] > 0)
        )
        univ = elig[elig].index
        if len(univ) < N_HOLD * 2:
            continue

        f_rev = zscore(-rev.loc[d, univ])
        f_size = zscore(-np.log(cap.loc[d, univ]))
        f_turn = zscore(-turn_ma.loc[d, univ])
        f_vol = zscore(-vol.loc[d, univ])

        score = (f_rev + f_size + f_turn + f_vol) / 4.0
        picks = score.nlargest(N_HOLD).index

        w = pd.Series(1.0 / len(picks), index=picks)

        # one-way turnover vs previous book
        allc = prev_w.index.union(w.index)
        turnovers.append(0.5 * (w.reindex(allc).fillna(0.0)
                                - prev_w.reindex(allc).fillna(0.0)).abs().sum())
        prev_w = w

        weight_rows.append(pd.DataFrame({"date": d, "code": w.index, "weight": w.values}))

    weights = pd.concat(weight_rows, ignore_index=True)

    # ---- backtest: weights set on rebalance date t apply from t+1 onward --
    # start the day after the FIRST date that actually produced a book (early
    # months get skipped by the MIN_HIST screen)
    first_rebal = weights["date"].min()
    start = pos_of[first_rebal] + 1
    hold_dates = dates[start:]

    # forward-fill the target book across the holding period
    # non-held names are 0 (not NaN) on a rebalance date, so the ffill below
    # replaces the whole book each month instead of accumulating past holdings
    w_wide = weights.pivot(index="date", columns="code", values="weight")
    w_wide = w_wide.reindex(columns=ret.columns).fillna(0.0)
    w_daily = w_wide.reindex(dates).shift(1).ffill().loc[hold_dates].fillna(0.0)

    r = ret.loc[hold_dates].fillna(0.0)
    port_ret = (w_daily * r).sum(axis=1)

    nav = (1.0 + port_ret).cumprod()
    nav = pd.concat([pd.Series([1.0], index=[dates[start - 1]]), nav])

    # ---- stats -----------------------------------------------------------
    ndays = len(port_ret)
    years = ndays / 252.0
    ann_ret = nav.iloc[-1] ** (1.0 / years) - 1.0
    ann_vol = port_ret.std(ddof=0) * np.sqrt(252.0)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan
    dd = nav / nav.cummax() - 1.0
    mdd = dd.min()

    # ---- outputs ---------------------------------------------------------
    used_codes = sorted(weights["code"].unique())
    prices = df[(df["date"] >= dates[start - 1]) & (df["code"].isin(used_codes))][
        ["date", "code", "close", "amount"]
    ].sort_values(["date", "code"])

    weights.to_csv(f"{OUTDIR}/weights.csv", index=False, date_format="%Y-%m-%d")
    nav.rename("nav").rename_axis("date").to_csv(f"{OUTDIR}/nav.csv",
                                                 date_format="%Y-%m-%d")
    prices.to_csv(f"{OUTDIR}/prices.csv", index=False, date_format="%Y-%m-%d")

    print("=== multi-factor long-only backtest ===")
    print(f"period            : {nav.index[0].date()} -> {nav.index[-1].date()} "
          f"({ndays} trading days, {years:.2f}y)")
    print(f"annualised return : {ann_ret:.2%}")
    print(f"annualised vol    : {ann_vol:.2%}")
    print(f"Sharpe (rf=0)     : {sharpe:.2f}")
    print(f"max drawdown      : {mdd:.2%}")
    print(f"final NAV         : {nav.iloc[-1]:.3f}")
    print(f"rebalance dates   : {weights['date'].nunique()}")
    print(f"distinct names    : {len(used_codes)}")
    print(f"avg one-way turnover per rebalance : {np.mean(turnovers[1:]):.1%}")
    print(f"rows -> weights {len(weights)}, nav {len(nav)}, prices {len(prices)}")


if __name__ == "__main__":
    main()
