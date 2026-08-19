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

★ 市场代理是本模块唯一的自造输入
------------------------------
工具只有价格面板，没有真市值加权指数，所以代理用价格面板股票池的
截面等权收益，且**逐标的留一**（算 r_i 的残差时代理不含 r_i）。
留一不是精致化，是正确性要求：共用一条含自己的均值会机械压出残差
负相关，把注数抬到超过持仓数（合成面板实测 20 只报 28.9 注）。
详见 `_loo_residuals` 的文档。

仍然存在的偏差：等权代理不等于用户真实的风险模型。用户若有自己的
因子残差，`residual_breadth()` 可以直接吃残差矩阵，绕开这一层。

同规模对照要求股票池至少是持仓数的 2 倍，否则抽出来的组合与本账
大量重叠、对照失去意义，此时该项跳过并写明原因。

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

# ne/b 要超出 1 多少才算「实质低估」而非估计噪声。
# ★ 注数是 126 天窗口上样本协方差的二次型之比，1~2% 的偏离本就在噪声里。
# 实测两次都栽在这里：demo 报「低估 0.98 倍」（比值 <1，语义上根本没低估）、
# 真实面板随机组合报「低估 1.01 倍」（纯噪声被写成低估）。
UNDERSTATE_NOISE = 0.05

# 本账注数落在同规模随机组合抽样分布的哪个分位（越高＝越比对照集中）。
# ★ 主判据用分位而不用 b/c 硬比值，理由与族三改用置换零分布相同：
# 硬门槛是刀锋判定。实测 6.5 注 vs 对照 11.8 注（比值 0.55）曾被 0.5 的
# 门槛判 OK 并写成「不是这本账的选股特征」，而 6.5 相对 11.8 已接近腰斩 ——
# 0.55 与 0.49 之间没有实质差别，却给出完全相反的对外说法。
CTRL_PCTILE_WARN = 0.80

# 比值门槛保留为兜底：分位算不出来时（对照抽样全退化）仍要有判据。
CTRL_RATIO_WARN = 0.5

# 本账注数低于这么高比例的同规模随机组合 ⇒ 压缩里有选股的贡献。
# ★ 分位判据优先于 CTRL_RATIO_WARN 那个比值门槛：比值 0.55 vs 0.49 之间
# 没有实质差别，却会给出完全相反的对外说法（实测真面板 0.55 被判 OK 并
# 写成「不是选股特征」，而 6.5 注 vs 11.8 注已接近腰斩）。
CTRL_PCTILE_WARN = 0.80

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


def residual_breadth(resid: np.ndarray, w: np.ndarray | None = None) -> float:
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

    ★ 和 residual_breadth() 不是同一个量，本模块把两个都算出来是为了让用户
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


MIN_COMPLEMENT = 10


def _residuals(block: pd.DataFrame, cols: list) -> tuple[np.ndarray, bool]:
    """去掉市场后的残差矩阵（T × len(cols)），市场代理取【持仓之外】的等权均值。

    返回 (残差矩阵, 代理是否干净)。

    ★ 代理的构造方式改过两轮，每轮都是合成面板抓出来的实质 bug。
    ----------------------------------------------------------
    第一版：整个股票池的等权均值，所有标的共用一条。
      每只标的自己就在那条均值里 ⇒ β_i 被机械地推向 1，残差之间压出约
      −1/(K−1) 的负相关。负相关让 w'Σw 小于对角线预测，注数被抬到
      【超过持仓数】：60 只池子持 20 只，实测 28.9 注，与解析预测
      20/(1 − 19/59) = 29.5 吻合。「20 只票持了 29 注独立的赌」荒谬 ——
      这个度量的上界本应是 1/Σw²。

    第二版（留一）：算 r_i 的残差时代理不含 r_i。
      修好了「无因子时 β 被推向 1」：独立面板上注数回到 19.98 ≈ 20。
      但真有共同因子时 β_i ≈ 1，此时 resid_i ≈ e_i − ē_{−i}，而 ē_{−i}
      含 e_j（权重 1/(K−1)）⇒ 残差仍被压出负相关，注数仍然超过持仓数
      （单因子面板实测 28.9）。⇒ 留一只修了一半。

    第三版（本版，持仓之外）：代理 = 股票池里【不在本账】的标的等权均值。
      代理与任何持仓标的都不共享特有项 ⇒ 不再有负相关通道。
      残余偏差改为【正】的一小项（所有持仓共享代理自己的特有噪声
      Var ≈ σ²/|complement|）⇒ 注数被【低估】、问题被【高报】。
      对审计工具这是安全方向（不会把有问题的账放过去），且量级随
      complement 变大而衰减，所以要求 complement ≥ MIN_COMPLEMENT。

    complement 不够时退回留一，并把这件事回报给上层写进报告 ——
    静默降级等于让用户以为用的是干净口径。
    """
    Xall = block.fillna(0.0).to_numpy(dtype=float)
    Xall = Xall - Xall.mean(axis=0)
    pos = {c: i for i, c in enumerate(block.columns)}
    idx = [pos[c] for c in cols]
    Y = Xall[:, idx]

    out = [i for i in range(Xall.shape[1]) if i not in set(idx)]
    if len(out) >= MIN_COMPLEMENT:
        m = Xall[:, out].mean(axis=1)
        m = m - m.mean()
        den = float(m @ m)
        if den <= 0:
            return Y, False
        beta = (Y.T @ m) / den
        return Y - np.outer(m, beta), True

    # 退路：留一代理（代理不含自己，但仍与其他持仓共享特有项）
    K = Xall.shape[1]
    if K < 3:
        return Y, False
    tot = Xall.sum(axis=1)
    M = (tot[:, None] - Y) / (K - 1)
    M = M - M.mean(axis=0)
    den_v = (M * M).sum(axis=0)
    num_v = (Y * M).sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        b = np.where(den_v > 0, num_v / den_v, 0.0)
    return Y - M * b, False


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
    # 市场代理在 _loo_residuals 里逐标的构造（留一），不在这里算一条共用的。
    # 共用一条会把标的自己算进代理，机械压出残差负相关，见该函数文档。
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
        # 有效日：该窗口内至少有两只标的有收益（留一代理才有东西可算）
        good = (blk.notna().sum(axis=1) >= 2).to_numpy()
        if good.sum() < min_obs:
            continue
        blk = blk[good]

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
        # ★ 代理必须从【整个股票池】construct，不能只用持仓。
        # 只用持仓时代理会随持仓数变化，本账和同规模对照就不是在同一个
        # 市场定义下比较了 —— 识别检验要求股票池固定、只换挑哪些名字。
        uni = list(enough)
        resid, clean = _residuals(blk[uni], held)

        rec = {
            "date": t,
            "n": len(held),
            "n_obs": int(good.sum()),
            "n_universe": int(len(enough)),
            "proxy_clean": bool(clean),
            "ne_nominal": effective_names(w),
            "breadth": residual_breadth(resid, w),
            "breadth_ew": residual_breadth(resid),
            "enb": enb(resid, w),
            "breadth_ctrl": np.nan,
        }

        # 同规模随机对照：股票池够大才做，否则抽出来的组合和本账
        # 重叠过半，对照失去意义。
        pool = list(enough)
        if len(pool) >= 2 * len(held):
            draws = []
            # ★ 对照必须用【和本账相同的权重向量】。
            # 第一版这里漏了 w，对照实际是等权，而本账是市值加权 ——
            # 于是「本账 7.0 注 vs 对照 30.7 注」里绝大部分差异来自
            # 加权方式，不是选股。报告自己就带着反证：同一批持仓改等权
            # 是 11.3 注（不是 7.0）。文档承诺「加权固定」，代码没做到。
            # 识别检验只允许变一个东西：挑哪些名字。
            for _ in range(N_CTRL_DRAWS):
                pick = list(rng.choice(pool, len(held), replace=False))
                r_ctrl, _ = _residuals(blk[uni], pick)
                draws.append(residual_breadth(r_ctrl, w))
            draws = [d for d in draws if np.isfinite(d)]
            if draws:
                rec["breadth_ctrl"] = float(np.median(draws))
                # ★ 20 次抽样本身就是一个零分布 —— 记下本账落在它的哪个分位。
                # 只报「比值 0.55」再拿 0.5 当门槛是刀锋判定：0.55 判「不是选股」、
                # 0.49 判「是选股」，而这个量本身带抽样噪声。分位数才是识别检验
                # 该用的判据（与族三的置换零分布同一套办法）。
                bk = rec["breadth"]
                if np.isfinite(bk):
                    rec["ctrl_pctile"] = float(np.mean(
                        [1.0 if bk < x else 0.0 for x in draws]))
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
    # ★ 代理噪声偏差必须报，不能让小股票池假装成告警。
    # 代理是 complement 的等权均值，本身带 σ²/C 的特有噪声，β≈1 时
    # 这份噪声进到每只持仓的残差里 ⇒ 压出正相关 ρ≈1/C ⇒ 注数被低估。
    # 解析预测 b ≈ ne/(1+(ne−1)/C)，反解得到去偏后的下界。
    comp = _median(d, "n_universe") - n if np.isfinite(_median(d, "n_universe")) else np.nan
    b_adj = np.nan
    if np.isfinite(comp) and comp > 0 and np.isfinite(ne) and ne > 1:
        # 由观测 b 反解「若代理无噪声」的注数：1/b_true = 1/b − (ne−1)/(ne·C)
        inv = 1.0 / b - (ne - 1.0) / (ne * comp)
        if inv > 0:
            # ★ 必须按 1/Σw² 截断。去偏是一阶近似，小 C 且 b 已接近 ne 时
            # 会反解出【超过持仓数】的注数（demo 实测 8 只报 12.9 注）——
            # 注数的硬上界就是名义有效持仓数，报出去就是自相矛盾。
            b_adj = float(min(1.0 / inv, ne))
    rep.stats["breadth_complement"] = comp
    rep.stats["breadth_proxy_adjusted"] = b_adj
    if np.isfinite(b_adj) and np.isfinite(b) and b > 0 and abs(b_adj - b) / b >= 0.10:
        extra.append(f"市场代理由股票池里非持仓的 {comp:.0f} 只构成，"
                     f"它自身的特有噪声会把注数压低；扣掉这一项约 "
                     f"{b_adj:.1f} 注（合成面板上该修正的解析预测与实测"
                     f"吻合到 2%）⇒ 下面的倍数按未扣的 {b:.1f} 报，是保守侧")
    if not bool(d.get("proxy_clean", pd.Series([True])).all()):
        extra.append(f"部分调仓日股票池里非持仓的标的不足 "
                     f"{MIN_COMPLEMENT} 只，那些日期退回了留一代理"
                     "（代理与其他持仓共享特有项，注数会被抬高）")
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

    # ★ 倍数只在实质大于 1 时才有意义，且要留估计噪声余量。
    # 实测两次翻车：demo 打印「低估 0.98 倍」（ratio<1，语义上等于没低估）；
    # 真实面板随机组合打印「低估 1.01 倍」（ratio=1.0，纯估计噪声却写成低估）。
    # 注数是 126 天窗口上的样本协方差之比，1~2% 的偏离本就在噪声里。
    if ratio > 1.0 + UNDERSTATE_NOISE:
        impact = (f"按「残差互不相关」编特定风险预算，会把组合残差波动低估 "
                  f"{vol_x:.2f} 倍（真实 w'Σw 是对角线预测的 {ratio:.1f} 倍）。"
                  f"\n★ 这不影响净值对不对，影响的是风险预算和对外披露的"
                  f"分散化程度")
    else:
        impact = (f"残差注数与名义有效持仓数实质相当（{b:.1f} vs {ne:.1f}，"
                  f"差异在 {UNDERSTATE_NOISE:.0%} 的估计噪声内）⇒ "
                  f"按残差独立编的特定风险预算【没有】被这本账的残差共动"
                  f"击穿，这一项无需打折")
    level = (WARN if ratio > 1.0 + UNDERSTATE_NOISE
             and vol_x >= UNDERSTATE_WARN else OK)
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

    # ★ 判据用【本账落在对照抽样分布的哪个分位】，不用 b/c 硬门槛。
    # 第一版拿 0.5 当门槛，于是真实面板上 6.5 注 vs 对照 11.8 注（比值 0.55）
    # 被判 OK 并写成「不是这本账的选股特征」—— 6.5 相对 11.8 已经接近腰斩，
    # 结论与自己给出的数字相反。0.55 与 0.49 之间没有实质差别，
    # 却导致完全相反的对外说法，这是刀锋判定（与族三改用置换零分布同因）。
    pct = _median(d, "ctrl_pctile") if "ctrl_pctile" in d.columns else np.nan
    if np.isfinite(pct):
        rep.stats["breadth_control_pctile"] = pct
        # ★ 分位是「本账比多少比例的对照更集中」，措辞不能写成
        # 「低于 100% 的同规模随机组合」—— 真实面板上 pct=1.0 时那句话
        # 读起来像「比所有组合都低，包括它自己」，是自相矛盾的。
        # 20 次抽样只能分辨到 1/20，所以 1.0 要报成「全部 20 次」。
        where = (f"比全部 {N_CTRL_DRAWS} 次抽样都更集中"
                 if pct >= 1.0 - 1e-9 else
                 f"比 {pct:.0%} 的同规模随机组合更集中")
        detail += f"\n本账注数{where}（{N_CTRL_DRAWS} 次抽样，各调仓日分位的中位数）"

    if (np.isfinite(pct) and pct >= CTRL_PCTILE_WARN) or r < CTRL_RATIO_WARN:
        if np.isfinite(pct):
            why = (f"比全部 {N_CTRL_DRAWS} 次同规模随机抽样都更集中"
                   f"（{b:.1f} 注 vs 对照 {c:.1f} 注）"
                   if pct >= 1.0 - 1e-9 else
                   f"比 {pct:.0%} 的同规模随机组合更集中")
        else:
            why = f"只有同规模随机组合的 {r:.0%}"
        rep.add(WARN, "同规模对照", detail,
                f"本账注数{why} ⇒ 这个压缩里有【选股】的贡献，"
                "不能全归给股票池和窗口。对外披露分散化程度时应按残差注数"
                "而非持仓只数，特定风险预算也要按上一项的低估倍数放大",
                section=SECTION)
    else:
        # ★ 不能无条件写「绝对低估倍数仍然成立」——上一项可能报的是
        # 「没有低估」。两项的口径必须对得上，否则报告自相矛盾。
        # 判据必须与上一项【同一条】：那边用 ne/b > 1+UNDERSTATE_NOISE 才算
        # 实质低估。这里若只判 >= 1.0，纯估计噪声也会让这句话出现，
        # 于是报告①说「无需打折」、②说「低估倍数仍然成立」——自相矛盾。
        _ne = rep.stats.get("breadth_ne_nominal", np.nan)
        _b = rep.stats.get("breadth_residual", np.nan)
        _substantive = (np.isfinite(_ne) and np.isfinite(_b) and _b > 0
                        and _ne / _b > 1.0 + UNDERSTATE_NOISE)
        tail = "上一项报的绝对低估倍数仍然成立，只是" if _substantive else ""
        rep.add(OK, "同规模对照", detail,
                f"本账注数是同规模随机组合的 {r:.0%}"
                + (f"、落在 {pct:.0%} 分位" if np.isfinite(pct) else "")
                + " ⇒ 与同规模随机组合不可区分，压缩主要来自股票池和窗口本身。"
                f"★ {tail}不该说成「我们的选股让注数塌了」",
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
