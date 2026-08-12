"""核心数学：期间收益、换手、缺失价格政策。

★ 本模块的每个函数都必须能被对账
------------------------------
实测教训：闸门本身也会标定错（连错四次之后才改用对账型断言）。
所以这里的收益口径给了两条独立实现：

    period_returns()  按调仓区间一次算清（买入持有）
    daily_path()      按日复利、权重随价格漂移

两者在数学上恒等（区间内买入持有）。测试断言两者偏差 < 1e-12。
不是为了好看 —— 是因为漂移权重的实现极易写错，而写错之后
净值仍然「看起来合理」。

★ 换手的正确口径
--------------
    naive       0.5 · Σ|w_t − w_{t−1}|              ← 常见但错
    drift-adj   0.5 · Σ|w_t − w̃_{t−1}|             ← 正确

其中 w̃ 是上期权重被区间收益推着漂移、再按 gross 归一后的结果。
朴素口径把「价格涨了导致权重变大」也算成一笔交易 —— 那笔交易
并不存在，你没有下单。方向：朴素口径【高估】换手。

而由信号自相关反推换手是【低估】：实测推 1.98x、实际 4.6x，
低估 2.6 倍。两个偏差方向相反，所以不能互相抵消着蒙。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# 缺失价格的三种记账政策。
# ★ 没有哪一种是「对」的 —— 正确做法是只在权威退市日挂损失。
# 工具的价值在于把三种政策的净值差给出来，作为记账不确定性的【区间】。
MISSING_POLICIES = ("drop", "hold_last", "zero")

TRADING_DAYS = 252


def price_matrix(p: pd.DataFrame) -> pd.DataFrame:
    """价格长表 → date × code 矩阵。不填充 —— 缺失就是缺失。"""
    m = p.pivot(index="date", columns="code", values="close")
    return m.reindex(sorted(m.columns), axis=1).sort_index()


def align(wm: pd.DataFrame, pm: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """把权重矩阵的列对齐到价格矩阵（并集，缺失列价格为 NaN）。"""
    cols = sorted(set(wm.columns) | set(pm.columns))
    return wm.reindex(columns=cols, fill_value=0.0), pm.reindex(columns=cols)


def _seg_gross(pm: pd.DataFrame, t0, t1) -> pd.Series:
    """区间 [t0, t1] 的毛增长因子 P(t1)/P(t0)。缺任一端为 NaN。"""
    if t0 not in pm.index or t1 not in pm.index:
        return pd.Series(np.nan, index=pm.columns)
    a, b = pm.loc[t0], pm.loc[t1]
    with np.errstate(divide="ignore", invalid="ignore"):
        g = b / a.replace(0.0, np.nan)
    return g


def _seg_last_gross(pm: pd.DataFrame, t0, t1) -> pd.Series:
    """区间内【最后一个可见价格】/ P(t0)。

    ★ 这个函数的存在是修一个实质 bug。
    hold_last 政策的语义是「按最后可见价格持有」，第一版却把缺价的
    毛增长直接设成 1.0 —— 那是「按【买入】价持有」，等于把退市前的
    崩盘整段丢掉：一只跌 80% 然后退市的票被记成 0% 收益。

    真实退市股在消失前会崩盘（ST → 面值退市，常见 −50%~−90%），
    所以这个差别不是细节：它决定了 drop / hold_last / zero 三种政策
    的高低次序是否有定向意义。
    """
    if t0 not in pm.index:
        return pd.Series(np.nan, index=pm.columns)
    seg = pm.loc[t0:t1]
    last = seg.ffill().iloc[-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        return last / pm.loc[t0].replace(0.0, np.nan)


def period_returns(wm: pd.DataFrame, pm: pd.DataFrame,
                   policy: str = "hold_last") -> pd.DataFrame:
    """每个调仓区间的组合收益（买入持有口径）。

    返回 DataFrame，index=调仓日 t_k，列：
        end          区间末日 t_{k+1}
        ret          区间组合收益
        w_missing    该期权重中价格缺失的部分（0~1）
        n_missing    缺失标的数
    """
    if policy not in MISSING_POLICIES:
        raise ValueError(f"policy 须为 {MISSING_POLICIES} 之一，收到 {policy!r}")

    wm, pm = align(wm, pm)
    dates = list(wm.index)
    rows = []
    for t0, t1 in zip(dates[:-1], dates[1:]):
        w = wm.loc[t0]
        held = w != 0.0
        g = _seg_gross(pm, t0, t1)
        bad = held & ~np.isfinite(g)
        w_missing = float(w[bad].abs().sum())

        if policy == "drop":
            # 无损移除：只在有价格的标的上重新归一
            # ★ 实测这会免费扔掉亏损，把净值抬高约 +9pp
            keep = held & np.isfinite(g)
            wk = w[keep]
            denom = float(wk.abs().sum())
            ret = float((wk * (g[keep] - 1.0)).sum() / denom) if denom > 0 else 0.0
        elif policy == "hold_last":
            # 按【最后可见价格】持有 —— 保留退市前的崩盘，只丢掉之后的未知段
            gl = _seg_last_gross(pm, t0, t1)
            gg = g.copy()
            gg[bad] = gl[bad]
            # 连一个可见价格都没有（整段缺失）⇒ 只能按不动处理
            gg[bad & ~np.isfinite(gg)] = 1.0
            ret = float((w[held] * (gg[held] - 1.0)).sum())
        else:  # zero
            # 全额清算为 0（−100%）
            # ★ 实测这会造出虚假暴跌，约 −37pp
            gg = g.copy()
            gg[bad] = 0.0
            ret = float((w[held] * (gg[held] - 1.0)).sum())

        rows.append(dict(end=t1, ret=ret, w_missing=w_missing,
                         n_missing=int(bad.sum())))
    return pd.DataFrame(rows, index=pd.Index(dates[:-1], name="date"))


def daily_path(wm: pd.DataFrame, pm: pd.DataFrame) -> pd.Series:
    """按日复利的组合收益序列，区间内权重随价格漂移。

    ★ 这是 period_returns(policy="hold_last") 的独立实现。
    两者恒等（区间内买入持有），用于对账。
    """
    wm, pm = align(wm, pm)
    reb = list(wm.index)
    out: dict = {}
    for t0, t1 in zip(reb[:-1], reb[1:]):
        days = [d for d in pm.index if t0 <= d <= t1]
        if len(days) < 2:
            continue
        w = wm.loc[t0].copy()
        held = w != 0.0
        cur = w[held].copy()
        for a, b in zip(days[:-1], days[1:]):
            pa, pb = pm.loc[a, cur.index], pm.loc[b, cur.index]
            with np.errstate(divide="ignore", invalid="ignore"):
                g = (pb / pa.replace(0.0, np.nan))
            g = g.where(np.isfinite(g), 1.0)          # 缺价 ⇒ 该日不动
            base = float(cur.abs().sum())
            grown = cur * g
            out[b] = float(grown.sum() - cur.sum()) / base if base > 0 else 0.0
            cur = grown
    return pd.Series(out).sort_index()


def drift_weights(w_prev: pd.Series, g: pd.Series) -> pd.Series:
    """上期权重被区间毛增长 g 推着漂移，再按 gross 归一。

    缺价的标的按 g=1 处理（不动），与 hold_last 政策一致。
    """
    gg = g.reindex(w_prev.index)
    gg = gg.where(np.isfinite(gg), 1.0)
    drifted = w_prev * gg
    gross = float(drifted.abs().sum())
    return drifted / gross if gross > 0 else drifted * 0.0


def turnover(wm: pd.DataFrame, pm: pd.DataFrame | None = None) -> pd.DataFrame:
    """逐期换手：朴素口径与漂移调整口径。

    返回 index=调仓日（从第 2 期起），列 naive / drift_adj。
    单边口径（0.5·Σ|Δw|）—— 双边成本请自行 ×2。
    """
    dates = list(wm.index)
    rows = []
    for prev, cur in zip(dates[:-1], dates[1:]):
        w0, w1 = wm.loc[prev], wm.loc[cur]
        naive = 0.5 * float((w1 - w0).abs().sum())
        if pm is None:
            adj = np.nan
        else:
            g = _seg_gross(pm, prev, cur)
            adj = 0.5 * float((w1 - drift_weights(w0, g)).abs().sum())
        rows.append(dict(naive=naive, drift_adj=adj))
    return pd.DataFrame(rows, index=pd.Index(dates[1:], name="date"))


def rank_autocorr_turnover(wm: pd.DataFrame) -> float:
    """由持仓权重的期间自相关反推换手（业界常见近似）。

    ★ 这个数会【低估】真实换手。实测：反推 1.98x / 实际 4.6x（低估 2.6 倍）。
    本函数存在的唯一目的是把这个低估量算出来给客户看，
    不是推荐用它估成本。
    """
    dates = list(wm.index)
    acs = []
    for prev, cur in zip(dates[:-1], dates[1:]):
        a, b = wm.loc[prev], wm.loc[cur]
        both = (a != 0) | (b != 0)
        if both.sum() < 3:
            continue
        ra = a[both].rank()
        rb = b[both].rank()
        if ra.std(ddof=0) == 0 or rb.std(ddof=0) == 0:
            continue
        acs.append(float(np.corrcoef(ra, rb)[0, 1]))
    if not acs:
        return float("nan")
    rho = float(np.mean(acs))
    # 常见近似：单边换手 ≈ (1 − ρ) / 2
    return max(0.0, (1.0 - rho) / 2.0)


def annualize(rets: pd.Series, periods_per_year: float) -> dict:
    """年化收益 / 波动 / Sharpe（rf=0）。"""
    r = pd.Series(rets).dropna().astype(float)
    if len(r) < 2:
        return dict(ann_ret=np.nan, ann_vol=np.nan, sharpe=np.nan, n=len(r))
    total = float((1.0 + r).prod())
    yrs = len(r) / periods_per_year
    ann_ret = total ** (1.0 / yrs) - 1.0 if yrs > 0 and total > 0 else np.nan
    sd = float(r.std(ddof=1))

    # ★ 不能只判 sd > 0。恒定收益序列的样本标准差是 ~1e-18 而非精确 0
    # （浮点），于是 `sd > 0` 通过，Sharpe 算出 9.8e15 这种荒谬值。
    # 测试就是这么抓到的。按相对尺度判「实质为零」才对。
    scale = max(abs(float(r.mean())), 1.0)
    degenerate = sd <= 1e-12 * scale

    ann_vol = 0.0 if degenerate else sd * np.sqrt(periods_per_year)
    sharpe = (np.nan if degenerate
              else float(r.mean()) / sd * np.sqrt(periods_per_year))
    return dict(ann_ret=ann_ret, ann_vol=ann_vol, sharpe=sharpe, n=len(r))


def periods_per_year(dates) -> float:
    """由调仓日间隔估每年期数。"""
    d = pd.Series(pd.to_datetime(pd.Index(dates))).sort_values()
    if len(d) < 3:
        return float(TRADING_DAYS)
    med = float(d.diff().dt.days.dropna().median())
    return 365.25 / med if med > 0 else float(TRADING_DAYS)
