"""族四：风险身份 —— 你报的持仓数不是独立注数。

审的是「这份组合到底持了几注独立的赌」，三项：

    ① 残差有效注数    名义有效持仓数 1/Σw² vs 去掉市场后的独立注数
    ② 同规模对照      这个压缩是你【选股】的性质，还是任何同规模组合都这样
    ③ 与 ENB 并列     和 Meucci(2009) 的有效注数是不是同一个量

★ 这一族为什么不出 BLOCK
----------------------
BLOCK 的语义是「净值不可信，先修」。注数压缩不会让净值算错 ——
它让【风险预算】算错：按残差独立编的特定风险预算会系统性低估
组合残差波动。所以本族最高只到 WARN，这是刻意的，不是漏了分级。

★ 度量本身：为什么是这个比值而不是领先特征值
--------------------------------------
    breadth = [w'diag(Σ)w] / [w'Σw] / Σwᵢ²

分子分母都是残差协方差 Σ 的二次型：分子是「假设残差互不相关」的方差预测，
分母是真实方差预测。残差真独立时两者相等，breadth = 1/Σwᵢ²
（等权即名义持仓数 N）；残差共动时分母变大，注数塌下来。

选它而不选领先特征值占比，是因为后者与平均两两相关系数相关 0.9985
（本度量的来源研究自己报的）—— 一个卖「一个特征值」而对手是
「一个平均数」的量没有可辩护的差异。这个比值回答的是平均相关系数
答不了的问题：市场模式被拿掉之后，这本账自己的残差还分散不分散。

★ 市场代理是本模块唯一的自造输入，偏差方向必须说清
----------------------------------------------
工具只有价格面板，没有真正的市值加权市场指数，所以市场代理用
【价格面板全体的截面等权收益】。当价格面板≈持仓本身时，这个代理
就是这本账自己的平均收益，减掉它等于把这本账的共同模式直接减掉 ⇒
残差看起来更独立 ⇒ **breadth 被高估、问题被低报**。

偏差方向是保守的（不会凭空造出告警），但仍然是偏差，所以：
面板股票池不到持仓数 2 倍时不跑同规模对照，不到持仓数+10 或 30 只
时整族跳过并写明原因。

★ 高维区间（n 接近或超过窗口长度）
--------------------------------
样本协方差在 n > T 时秩不足。breadth 是二次型之比、不需要求逆，
所以仍可算；但估计噪声随 n/T 上升。n/T > 0.5 时报告里会点出来，
不静默。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .core import align
from .report import OK, WARN, AuditReport

SECTION = "风险身份"

# 残差协方差的估计窗口（交易日）与该窗口内每只标的的最少观测数
WINDOW = 126
MIN_OBS = 100

# 少于这么多只标的时注数没有解释力（一只票按定义就是一注）
MIN_NAMES = 5

# 最多测这么多个日期。★ 超出会抽样，且抽样必须写进报告 ——
# 静默截断会被读成「全都测过了」。
MAX_DATES = 60

# 同规模随机对照的抽样次数与固定种子（同一份输入必须给同一份报告）
N_CTRL_DRAWS = 20
CTRL_SEED = 20260812

# 残差波动低估倍数：按残差独立编预算，实际波动是预算的多少倍
UNDERSTATE_WARN = 2.0

# 本账注数 / 同规模随机组合注数。低于此值 ⇒ 压缩是选股带来的，
# 不是「同规模组合都这样」
CTRL_RATIO_WARN = 0.5

# n/T 超过这个比例时在报告里点明高维估计噪声
HIGHDIM_NOTE = 0.5


def effective_names(w: np.ndarray) -> float:
    """名义有效持仓数 1/Σwᵢ²。

    这是权重自己就能算出来的数 —— 说明书上「持仓 130 只」隐含的
    分散化主张用的就是它（等权时恰好等于只数）。本模块的全部工作
    就是把它和【残差】层面的真实注数摆在一起。
    """
    s = float(np.sum(np.asarray(w, dtype=float) ** 2))
    return 1.0 / s if s > 0 else np.nan


def breadth(resid: np.ndarray, w: np.ndarray | None = None) -> float:
    """残差有效注数 = [w'diag(Σ)w] / [w'Σw] / Σwᵢ²。

    resid  T × n 残差矩阵（已去市场）
    w      组合权重，None 表示等权

    残差互不相关时返回 1/Σwᵢ²（等权即 n）；完全共动时趋于 1。
    """
    resid = np.asarray(resid, dtype=float)
    n = resid.shape[1]
    # np.cov 对单列会塌成标量，之后的二次型就崩了。
    # 一只票按定义就是一注，直接给 nan 让上层跳过。
    if n < 2 or resid.shape[0] < 3:
        return np.nan
    if w is None:
        w = np.full(n, 1.0 / n)
    w = np.asarray(w, dtype=float)
    S = np.cov(resid, rowvar=False)
    full = float(w @ S @ w)
    diag = float(w @ np.diag(np.diag(S)) @ w)
    if not np.isfinite(full) or not np.isfinite(diag) or full <= 0 or diag <= 0:
        return np.nan
    ne = effective_names(w)
    return diag / full * ne if np.isfinite(ne) else np.nan


def enb(resid: np.ndarray, w: np.ndarray | None = None) -> float:
    """Meucci(2009) 有效注数：残差主成分上方差贡献的熵指数。

    ★ 和 breadth() 不是同一个量，本模块把两个都算出来是为了让用户
    自己核对，而不是由我们断言「不一样」。

    ENB 问的是风险在互不相关的方向上摊得均不均匀；breadth 问的是
    「假设独立」的方差预测偏了多少倍。后者能和说明书上的持仓数直接
    比较，前者没有对应的计数基准。
    """
    resid = np.asarray(resid, dtype=float)
    n = resid.shape[1]
    if n < 2 or resid.shape[0] < 3:
        return np.nan
    if w is None:
        w = np.full(n, 1.0 / n)
    w = np.asarray(w, dtype=float)
    S = np.cov(resid, rowvar=False)
    ev, V = np.linalg.eigh(S)
    ev, V = ev[::-1], V[:, ::-1]
    contrib = (V.T @ w) ** 2 * ev
    tot = float(contrib.sum())
    if not np.isfinite(tot) or tot <= 0:
        return np.nan
    pk = np.clip(contrib / tot, 1e-15, None)
    return float(np.exp(-np.sum(pk * np.log(pk))))


def _returns(pm: pd.DataFrame) -> pd.DataFrame:
    """价格矩阵 → 日收益矩阵。

    手算而不用 pct_change：后者的 fill_method 默认值在 pandas 各版本
    之间变过（1.x 默认前向填充），会把停牌日静默填成 0 收益。
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        r = pm / pm.shift(1) - 1.0
    return r.where(np.isfinite(r))


def _residuals(X: pd.DataFrame, mc: np.ndarray) -> np.ndarray:
    """对去均值的市场代理 mc 做单因子回归，返回残差矩阵。"""
    Xc = X.fillna(0.0).to_numpy(dtype=float)
    Xc = Xc - Xc.mean(axis=0)
    den = float(mc @ mc)
    if den <= 0:
        return Xc
    beta = (Xc.T @ mc) / den
    return Xc - np.outer(mc, beta)


def residual_breadth_panel(wm: pd.DataFrame, pm: pd.DataFrame,
                           window: int = WINDOW, min_obs: int = MIN_OBS,
                           max_dates: int = MAX_DATES) -> tuple[pd.DataFrame, list[str]]:
    """逐调仓日算残差注数、同规模对照与 ENB。

    三项检查共用这一份残差矩阵 —— 分三次算等于对面板扫三遍。

    返回 (逐日结果表, 说明/限制清单)。表列：
        n            本期实际参与计算的持仓数
        n_obs        窗口内有效观测天数
        n_universe   面板股票池里够观测数的标的数
        ne_nominal   名义有效持仓数 1/Σw²
        breadth      按【你的实际权重】的残差有效注数
        breadth_ew   同一批标的改成等权（权重敏感性）
        breadth_ctrl 同规模随机组合的注数中位数（识别对照）
        enb          同一残差矩阵上的 Meucci ENB
    """
    notes: list[str] = []
    wm, pm = align(wm, pm)
    rm = _returns(pm)
    # 市场代理：面板全体的截面等权收益。偏差方向见模块文档。
    mkt = rm.mean(axis=1, skipna=True)

    dates = [d for d in wm.index if d in rm.index]
    if len(dates) > max_dates:
        step = int(np.ceil(len(dates) / max_dates))
        kept = dates[::step]
        notes.append(f"调仓日共 {len(dates)} 个，每 {step} 个取 1 个"
                     f"（实测 {len(kept)} 个）以控制计算量")
        dates = kept

    rng = np.random.default_rng(CTRL_SEED)
    rows = []
    for t in dates:
        loc = rm.index.get_loc(t)
        if loc < window:
            continue
        sl = slice(loc - window + 1, loc + 1)
        blk = rm.iloc[sl]
        mk = mkt.iloc[sl]
        good = mk.notna().to_numpy()
        if good.sum() < min_obs:
            continue
        blk = blk[good]
        mkv = mk[good].to_numpy(dtype=float)
        mc = mkv - mkv.mean()

        enough = blk.columns[blk.notna().sum() >= min_obs]
        w_all = wm.loc[t]
        held = [c for c in w_all.index[w_all != 0.0] if c in set(enough)]
        if len(held) < MIN_NAMES:
            continue

        w = w_all[held].to_numpy(dtype=float)
        gross = float(np.abs(w).sum())
        if gross <= 0:
            continue
        w = w / gross
        resid = _residuals(blk[held], mc)

        rec = {
            "date": t,
            "n": len(held),
            "n_obs": int(good.sum()),
            "n_universe": int(len(enough)),
            "ne_nominal": effective_names(w),
            "breadth": breadth(resid, w),
            "breadth_ew": breadth(resid),
            "enb": enb(resid, w),
            "breadth_ctrl": np.nan,
        }

        # 同规模随机对照：股票池够大才做，否则抽出来的组合和本账
        # 重叠过半，对照失去意义。
        pool = list(enough)
        if len(pool) >= 2 * len(held):
            draws = []
            for _ in range(N_CTRL_DRAWS):
                pick = list(rng.choice(pool, len(held), replace=False))
                draws.append(breadth(_residuals(blk[pick], mc)))
            draws = [d for d in draws if np.isfinite(d)]
            if draws:
                rec["breadth_ctrl"] = float(np.median(draws))
        rows.append(rec)

    if not rows:
        return pd.DataFrame(), notes
    return pd.DataFrame(rows).set_index("date"), notes


def _median(d: pd.DataFrame, col: str) -> float:
    if col not in d.columns:
        return np.nan
    v = d[col].dropna()
    return float(v.median()) if len(v) else np.nan


def check_residual_breadth(d: pd.DataFrame, rep: AuditReport,
                           notes: list[str] | None = None) -> None:
    """① 名义有效持仓数 vs 残差有效注数，以及残差波动的低估倍数。"""
    if d is None or not len(d):
        rep.skip("残差有效注数", "没有任何调仓日凑齐窗口内的有效观测")
        return

    ne = _median(d, "ne_nominal")
    b = _median(d, "breadth")
    b_ew = _median(d, "breadth_ew")
    n = _median(d, "n")
    rep.stats["breadth_ne_nominal"] = ne
    rep.stats["breadth_residual"] = b
    rep.stats["breadth_residual_ew"] = b_ew
    rep.stats["breadth_n_dates"] = int(len(d))

    if not np.isfinite(b) or not np.isfinite(ne) or b <= 0:
        rep.skip("残差有效注数", "残差协方差退化，二次型算不出正值")
        return

    # 按残差独立编预算时，真实残差波动是预算的多少倍
    ratio = ne / b
    vol_x = float(np.sqrt(ratio)) if ratio > 0 else np.nan
    rep.stats["breadth_vol_understate_x"] = vol_x

    extra = []
    n_obs = _median(d, "n_obs")
    if np.isfinite(n) and np.isfinite(n_obs) and n_obs > 0 and n / n_obs > HIGHDIM_NOTE:
        extra.append(f"n/T = {n:.0f}/{n_obs:.0f} = {n / n_obs:.2f}，"
                     "处于高维区间，注数估计噪声偏大（度量本身仍成立，"
                     "它是二次型之比、不需要求逆）")
    if np.isfinite(b_ew) and np.isfinite(b) and b > 0:
        gap = abs(b_ew - b) / b
        if gap >= 0.15:
            extra.append(f"同一批标的改成等权是 {b_ew:.1f} 注"
                         f"（差 {gap:.0%}）⇒ 注数对权重敏感，"
                         "换加权方式会改变这个结论")
    for nt in (notes or []):
        extra.append(nt)

    detail = (f"名义有效持仓数 1/Σw² = {ne:.1f}"
              + (f"（{n:.0f} 只标的）" if np.isfinite(n) else "")
              + f"，去掉市场后的残差有效注数 = {b:.1f}"
              f"（{len(d)} 个调仓日的中位数）")
    if extra:
        detail += "\n" + "\n".join(extra)

    impact = (f"按「残差互不相关」编特定风险预算，会把组合残差波动低估 "
              f"{vol_x:.2f} 倍（真实 w'Σw 是对角线预测的 {ratio:.1f} 倍）。"
              f"\n★ 这不影响净值对不对，影响的是风险预算和对外披露的"
              f"分散化程度")
    level = WARN if vol_x >= UNDERSTATE_WARN else OK
    rep.add(level, "残差有效注数", detail, impact, section=SECTION)


def check_breadth_control(d: pd.DataFrame, rep: AuditReport) -> None:
    """② 识别检验：压缩是你选股的性质，还是任何同规模组合都这样。

    ★ 这一项是整族的承重墙。如果任何同规模等权组合都只有这么几注，
    那「6 注」就只是在数名字，没有任何需要解释的东西。所以对照必须
    固定股票池、窗口、加权、估计器，只换「挑哪些名字」。
    """
    if d is None or not len(d) or "breadth_ctrl" not in d.columns:
        rep.skip("同规模对照", "没有可用的注数结果")
        return
    ok = d[["breadth", "breadth_ctrl"]].dropna()
    if not len(ok):
        rep.skip("同规模对照",
                 "面板股票池不足持仓数的 2 倍，随机对照会与本账大量重叠")
        return

    b = float(ok["breadth"].median())
    c = float(ok["breadth_ctrl"].median())
    rep.stats["breadth_control"] = c
    rep.stats["breadth_control_n_dates"] = int(len(ok))
    if not np.isfinite(c) or c <= 0:
        rep.skip("同规模对照", "对照组的残差协方差退化")
        return

    r = b / c
    rep.stats["breadth_control_ratio"] = r
    detail = (f"本账 {b:.1f} 注 vs 同规模随机组合 {c:.1f} 注"
              f"（{N_CTRL_DRAWS} 次抽样的中位数，{len(ok)} 个调仓日）"
              f"\n股票池、窗口、加权、估计器全部固定，只换「挑哪些名字」")
    if r < CTRL_RATIO_WARN:
        rep.add(WARN, "同规模对照", detail,
                f"本账注数只有同规模随机组合的 {r:.0%} ⇒ 压缩是【选股】"
                f"带来的，不是「同规模组合都这样」。这个数可以对外披露，"
                f"因为它排除了「只是在数名字」这个解释",
                section=SECTION)
    else:
        rep.add(OK, "同规模对照", detail,
                f"本账注数是同规模随机组合的 {r:.0%} ⇒ 压缩主要来自"
                f"股票池和窗口本身，不是这本账的选股特征。"
                f"★ 绝对低估倍数仍然成立（见上一项），只是不该说成"
                f"「我们的选股让注数塌了」",
                section=SECTION)


def check_breadth_vs_enb(d: pd.DataFrame, rep: AuditReport) -> None:
    """③ 和 Meucci ENB 并列报告，不断言差异。"""
    if d is None or not len(d):
        rep.skip("与 ENB 并列", "没有可用的注数结果")
        return
    ok = d[["breadth", "enb"]].dropna()
    if len(ok) < 3:
        rep.skip("与 ENB 并列", f"只有 {len(ok)} 个调仓日两个量都算出来了，"
                               "不足以报相关性")
        return
    b = float(ok["breadth"].median())
    e = float(ok["enb"].median())
    corr = float(ok["breadth"].corr(ok["enb"]))
    rep.stats["breadth_enb"] = e
    rep.stats["breadth_enb_corr"] = corr
    rep.add(OK, "与 ENB 并列",
            f"同一批残差矩阵上：本度量 {b:.1f} 注、Meucci ENB {e:.1f}，"
            f"两者跨调仓日相关 {corr:+.2f}"
            f"\nENB 问「风险在不相关方向上摊得均不均」，本度量问"
            f"「假设独立的方差预测偏了几倍」",
            "两个都对，但不可互换：ENB 没有可以和说明书持仓数直接比较的"
            "计数基准，本度量有（残差独立时恰好等于 1/Σw²）",
            section=SECTION)
