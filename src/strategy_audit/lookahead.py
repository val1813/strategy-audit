"""族二：前视与成分变动记账。

审的是「这份净值有没有用到调仓时点拿不到的信息」，四项：

    ① 调仓日对齐    权重日的收益是否已被计入（用当日收盘定权重又吃当日收益）
    ② 权重前视      权重与同期收益的相关（vw 权重用含当期收益的市值最典型）
    ③ 股票池生存者  持仓是否只落在「活到末日」的标的上
    ④ 成分变动记账  缺价标的的三种政策给出净值区间

★ 前视要查三处，不是一处
----------------------
实测教训：前视门必须查 signal / vw 权重 / 股票池筛选三处。
只查信号滞后是最常见的漏口 —— 信号确实滞后了一天，但权重用的是
月末市值（含当月收益），股票池又是「当前仍在册」的名单。
三处里任何一处漏了，净值就偷了分。

★ 成分变动 ≠ 退市
----------------
实测过的两个错法：
    无损移除持仓  = 免费扔掉亏损 ⇒ 净值虚高约 +9pp
    全按退市清算  = 虚假暴跌     ⇒ 净值虚低约 −37pp
正确做法是只在【权威退市日】挂损失。工具无从得知权威日期，
所以给出三种政策的净值区间，作为记账不确定性的量化下界。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .core import align, period_returns, price_matrix
from .report import BLOCK, OK, WARN, AuditReport

SECTION = "前视与记账"

# 权重与同期收益的截面相关。
# ★ 这两个数是【效应量分级】，不是检出门槛 —— 检出看 t（见 T_SIGNIF）。
# 实测教训：真实面板上 corr=+0.088 的 vw 前视把毛年化抬了近 4 倍，
# 用 |corr|≥0.10 当检出门槛会把它放过去。
W_RET_CORR_WARN = 0.10       # 方差退化时的兜底幅度门槛
W_RET_CORR_BLOCK = 0.25      # 超过此幅度且显著 ⇒ BLOCK 而非 WARN

# 跨期一致性的显著性门槛（配对 t）
T_SIGNIF = 2.0

# 滞后一日后收益衰减超过这个比例 ⇒ 结论依赖当日不可得的信息
LAG_DECAY_WARN = 0.30
LAG_DECAY_BLOCK = 0.60

# 三种记账政策的净值差超过这个比例 ⇒ 记账口径本身就决定了结论
POLICY_SPREAD_WARN = 0.05
POLICY_SPREAD_BLOCK = 0.20


def check_rebalance_alignment(wm: pd.DataFrame, pm: pd.DataFrame,
                              rep: AuditReport) -> None:
    """① 调仓日当天的收益有没有被吃掉。

    正确口径：t 日收盘定权重 ⇒ 收益从 t 日收盘算到 t+1。
    错误口径：t 日收盘定权重 ⇒ 却把 t 日当天的涨幅也算进去。

    做法：把权重整体前移一日（用 t−1 的价格起点算收益），看收益是否
    显著变差。若变差很多，说明原口径吃了定权重当日的收益。
    """
    wm, pm = align(wm, pm)
    days = list(pm.index)
    pos = {d: i for i, d in enumerate(days)}

    base = period_returns(wm, pm)["ret"]
    if len(base) < 6:
        rep.skip("调仓日对齐", "调仓期数不足 6 期")
        return

    # 构造「吃掉当日收益」的口径：起点提前一个交易日
    shifted = {}
    for t in wm.index:
        i = pos.get(t)
        if i is None or i == 0:
            continue
        shifted[days[i - 1]] = wm.loc[t].values
    if len(shifted) < 6:
        rep.skip("调仓日对齐", "无法构造前移一日的对照")
        return
    wm_early = pd.DataFrame(shifted, index=wm.columns).T.sort_index()
    early = period_returns(wm_early, pm)["ret"]

    m_base, m_early = float(base.mean()), float(early.mean())
    rep.stats["ret_mean_asis"] = m_base
    rep.stats["ret_mean_eat_day0"] = m_early
    gain = m_early - m_base

    # ★ 必须按【配对差】看噪声，不能只看均值比。
    # 实测教训：10 只等权组合日波动约 0.8%，35 期配对差的标准误约 0.19%，
    # 所以「相对多 20%」在小组合上完全可能是 1.5σ 的噪声。
    # 只用比例做门 ⇒ 干净组合也会被报 WARN（第一版就是如此）。
    n = min(len(base), len(early))
    diff = pd.Series(early.values[:n]) - pd.Series(base.values[:n])
    se = float(diff.std(ddof=1) / np.sqrt(n)) if n > 2 else np.nan
    t_stat = float(diff.mean() / se) if se and np.isfinite(se) and se > 0 else np.nan
    # 最小可检出效应：本样本量下能识别的最小每期差异
    mde = 1.96 * se if np.isfinite(se) else np.nan
    rep.stats["day0_t"] = t_stat
    rep.stats["day0_mde"] = mde

    if m_base <= 0 or gain <= 0:
        rep.add(OK, "调仓日对齐",
                f"把起点提前一日（=吃掉定权重当日收益）平均每期 "
                f"{m_early:+.3%} vs 当前 {m_base:+.3%} —— 无额外收益，"
                "当前口径没有从调仓日当天取分",
                section=SECTION)
        return

    rel = gain / abs(m_base)
    if not np.isfinite(t_stat) or abs(t_stat) < 2.0:
        rep.add(OK, "调仓日对齐",
                f"吃掉定权重当日收益平均每期多 {gain:+.3%}（相对 {rel:+.1%}），"
                f"但 t={t_stat:.2f} 不显著"
                + (f"，本样本最小可检出 {mde:.3%}/期" if np.isfinite(mde) else "")
                + " —— 与噪声不可区分",
                section=SECTION)
        return

    lvl = BLOCK if (rel > 0.50 and abs(t_stat) >= 3.0) else WARN
    rep.add(lvl, "调仓日当天收益值 %.0f%%" % (rel * 100),
            f"若把区间起点提前一日（等价于用 t 日收盘定权重、又计入 t 日涨幅），"
            f"平均每期收益从 {m_base:+.3%} 升到 {m_early:+.3%}"
            f"（+{rel:.1%}，配对 t={t_stat:.2f}）",
            "这一段是你【必须确认没有拿到】的收益。"
            "若你的权重是按 t 日收盘价/因子算的，收益就只能从 t 日收盘起算。"
            "本项不能证明你错了，只能告诉你这个口径值多少 —— 请自查下单时点",
            section=SECTION)


def check_weight_lookahead(wm: pd.DataFrame, pm: pd.DataFrame,
                           rep: AuditReport) -> None:
    """② 权重是否与同期收益相关（vw 权重前视最典型）。

    ★ 实测漏口：月末市值加权，而「月末 cap」含了当月收益。
    这会让涨得多的票在【本期】拿到更大权重 —— 权重里含当期结果。
    检查：每期算 权重 与 该期收益 的截面秩相关，看是否系统性为正。
    """
    wm, pm = align(wm, pm)
    dates = list(wm.index)
    if len(dates) < 7:
        rep.skip("权重前视", "调仓期数不足 7 期")
        return

    corrs = []
    n_flat = 0          # 该期权重无截面变化（等权）⇒ 相关系数无定义
    n_short = 0         # 该期持仓/价格不足
    for t0, t1 in zip(dates[:-1], dates[1:]):
        if t0 not in pm.index or t1 not in pm.index:
            n_short += 1
            continue
        w = wm.loc[t0]
        held = w != 0.0
        if int(held.sum()) < 5:
            n_short += 1
            continue
        with np.errstate(divide="ignore", invalid="ignore"):
            r = pm.loc[t1, held.index[held]] / pm.loc[t0, held.index[held]] - 1.0
        ww = w[held]
        ok = np.isfinite(r) & np.isfinite(ww)
        if int(ok.sum()) < 5:
            n_short += 1
            continue
        rw, rr = ww[ok].rank(), r[ok].rank()
        if rw.std(ddof=0) == 0:
            n_flat += 1
            continue
        if rr.std(ddof=0) == 0:
            n_short += 1
            continue
        corrs.append(float(np.corrcoef(rw, rr)[0, 1]))

    if len(corrs) < 5:
        # ★ 跳过原因必须说对。等权组合的权重没有截面变化，
        # 相关系数【在数学上无定义】—— 那不是「持仓过少」。
        # 报错原因会让客户去修一个不存在的问题（factor-audit 实测教训：
        # 把复牌报成复权错误，客户从此不信后续所有告警）。
        if n_flat >= max(1, len(dates) // 2):
            rep.add(OK, "权重前视",
                    f"{n_flat} 期为等权（权重无截面变化）⇒ "
                    "权重与收益的相关系数无定义。"
                    "等权本身不可能携带当期收益信息，这一项对该组合不适用",
                    section=SECTION)
        else:
            rep.skip("权重前视",
                     f"有效期数不足 5 期（等权 {n_flat} 期、"
                     f"持仓或价格不足 {n_short} 期）")
        return

    c = pd.Series(corrs)
    mean_c = float(c.mean())
    sd = float(c.std(ddof=1))

    # ★ 期间方差为 0 意味着【各期完全一致】—— 那是最强的证据，不是缺证据。
    # 第一版写成 `std>0 else nan`，再用 `not isfinite(t) ⇒ OK`，
    # 于是 corr=+1.000、100% 期为正的植入前视被报成「无系统性关联」。
    # 方差退化必须按「一致性满分」处理，绝不能落进通过分支。
    degenerate = sd <= 1e-12 * max(abs(mean_c), 1e-6)
    t_stat = np.inf if degenerate and mean_c != 0 else (
        mean_c / (sd / np.sqrt(len(c))) if sd > 0 else np.nan)
    rep.stats["weight_ret_corr"] = mean_c
    rep.stats["weight_ret_corr_t"] = t_stat

    frac_pos = float((c > 0).mean())
    t_s = (f"各期相关系数完全一致（无期间方差）" if degenerate
           else "t=∞" if not np.isfinite(t_stat) else f"t={t_stat:.2f}")
    detail = (f"权重与【同期】收益的截面秩相关平均 {mean_c:+.3f}"
              f"（{len(c)} 期，{frac_pos:.0%} 期为正，{t_s}）")

    # ★ 判据是【显著性】而不是相关系数的绝对大小。
    # 真实面板实测（800 只 A 股 / 126 期 / 30 只持仓）：把 cap 换成【下期】
    # 市值这个教科书级 vw 前视，毛年化从 3.63% 抬到 13.85%（近 4 倍），
    # 而截面秩相关只有 +0.088 —— 第一版要求 |corr|≥0.10 才报，于是
    # t=4.15 的铁证被绝对幅度门槛挡在门外，判了 OK。
    # 30 只持仓上真前视就是「小而一致」，一致性才是证据，幅度只是效应量。
    significant = (abs(mean_c) >= W_RET_CORR_WARN if degenerate
                   else np.isfinite(t_stat) and abs(t_stat) >= T_SIGNIF)
    if not significant:
        # 不显著时才看幅度：幅度也小 ⇒ 确实没有关联
        rep.add(OK, "权重前视", detail + " —— 无系统性关联", section=SECTION)
        return

    # 显著为负：权重系统性偏向【跌】的票。那不是前视（前视会偏向涨的），
    # 更可能是策略本身的构造特征（如逆势加仓），报 OK 但说明白。
    if mean_c < 0:
        rep.add(OK, "权重前视",
                detail + " —— 系统性偏【负】，与前视方向相反"
                "（前视会让权重偏向当期涨的票）",
                section=SECTION)
        return

    lvl = BLOCK if mean_c > W_RET_CORR_BLOCK else WARN
    rep.add(lvl, "权重与同期收益相关", detail,
            "权重里可能含了本期结果。最常见的成因是市值加权用了"
            "【期末】市值（含当期收益）而非期初市值 —— "
            "涨得多的票因此在本期拿到更大权重。"
            "★ 幅度小不等于影响小：实测 800 只 A 股面板上 corr 仅 +0.088"
            "（t=4.15）的 vw 前视，把毛年化从 3.63% 抬到 13.85%。"
            "请确认权重所用的一切输入都截止到调仓日之前",
            section=SECTION)


def check_universe_survivorship(wm: pd.DataFrame, pm: pd.DataFrame,
                               rep: AuditReport, delisted=None) -> None:
    """③ 股票池筛选的生存者偏差。

    ★ 这是第三处前视，最容易漏：信号滞后了、权重也用期初市值了，
    但股票池是「今天仍在册」的名单 —— 等于用未来信息做了筛选。
    """
    last = pm.index.max()
    held = set(c for c in wm.columns if (wm[c] != 0).any())
    if not held:
        rep.skip("股票池生存者偏差", "权重全为 0")
        return
    if delisted is not None:
        dl = set(map(str, delisted))
        held_s = set(map(str, held))
        touched = held_s & dl
        rep.stats["n_held"] = len(held)
        rep.stats["n_held_dead"] = len(touched)
        if dl and not touched:
            rep.add(BLOCK, "持仓只落在存活标的上（生存者偏差）",
                    f"【事实：用户提供退市名单】名单含 {len(dl)} 只退市标的，"
                    "持仓从未包含其中任何一只",
                    "股票池可能使用了事后仍在册名单，请核对每个历史时点的可选池",
                    section=SECTION)
        else:
            rep.add(OK, "股票池包含退出标的",
                    f"【事实：用户提供退市名单】持有过 {len(touched)}/{len(dl)} 只名单内标的",
                    section=SECTION)
        return
    last_row = pm.loc[last]
    n_alive = int(last_row.notna().sum())
    n_cols = int(len(last_row))
    last_cov = n_alive / n_cols if n_cols else 0.0
    rep.stats["survivorship_last_coverage"] = last_cov
    if last_cov < 0.5:
        rep.skip("股票池生存者偏差",
                 f"末日价格覆盖率仅 {last_cov:.1%}（{n_alive}/{n_cols}）—— "
                 "面板过于稀疏，无法区分「已退市」与「面板没这天的数」")
        return
    alive = set(pm.loc[[last]].dropna(axis=1).columns)
    dead = held - alive
    frac_dead = len(dead) / len(held)
    rep.stats["n_held"] = len(held)
    rep.stats["n_held_dead"] = len(dead)

    # 价格面板里存在但从未被持有的「已死」标的
    pm_dead = set(pm.columns) - alive
    if not dead and pm_dead:
        rep.add(BLOCK, "持仓只落在存活标的上（生存者偏差）",
                f"持有过 {len(held)} 只，全部活到末日 {last.date()}；"
                f"而价格面板里有 {len(pm_dead)} 只已消失的标的从未被持有",
                "股票池是「事后仍在册」的名单 ⇒ 用未来信息做了筛选。"
                "factor-audit 实测：生存者偏差让 IC 只抬高 5.9%，"
                "但组合毛收益抬高 45.88pp（把年化 −50% 显示成 −4.6%）。"
                "这是前视的第三处，信号滞后与权重口径都对也不能免除",
                section=SECTION)
        return

    if not dead and not pm_dead:
        rep.add(WARN, "无法判定股票池生存者偏差",
                f"持有过 {len(held)} 只全部存活，但价格面板本身也不含任何"
                f"已消失标的",
                "价格面板是纯生存者面板，所以看不出股票池的问题。"
                "请先用 factor-audit 的面板体检确认数据含退市股",
                section=SECTION)
        return

    if frac_dead < 0.02:
        rep.add(WARN, "已退出标的占比偏低",
                f"持有过的 {len(held)} 只里仅 {len(dead)} 只"
                f"（{frac_dead:.1%}）已消失",
                "A 股近年年均退市率约 1%，多年样本应更高。"
                "股票池可能只含部分已退市标的", section=SECTION)
        return

    rep.add(OK, "股票池生存者偏差",
            f"持有过 {len(held)} 只，其中 {len(dead)} 只（{frac_dead:.1%}）"
            f"在末日已不在册 —— 股票池含已退出标的", section=SECTION)


def _classify_missing(wm: pd.DataFrame, pm: pd.DataFrame) -> tuple[int, int]:
    """把「持有但区间末无价格」的事件分成停牌与永久消失两类。

    返回 (n_resume, n_permanent)，单位是【标的-期】事件数。

    判据很朴素但可靠：区间末之后还能看到价格 ⇒ 停牌复牌；再也看不到 ⇒
    永久消失（疑似退市，仍需权威退市日确认）。面板自己就带着这个答案，
    第一版却没去问 —— 于是把 100% 的停牌报成了退市记账问题。
    """
    wm, pm = align(wm, pm)
    dates = list(wm.index)
    # 每只标的最后一个可见价格的日期
    last_seen = {}
    for c in pm.columns:
        s = pm[c].dropna()
        last_seen[c] = s.index.max() if len(s) else None

    n_resume = n_perm = 0
    for t0, t1 in zip(dates[:-1], dates[1:]):
        if t1 not in pm.index:
            continue
        w = wm.loc[t0]
        for c in w[w != 0].index:
            if np.isfinite(pm.loc[t1, c]):
                continue
            ls = last_seen.get(c)
            if ls is not None and ls > t1:
                n_resume += 1
            else:
                n_perm += 1
    return n_resume, n_perm


def check_membership_accounting(wm: pd.DataFrame, pm: pd.DataFrame,
                                rep: AuditReport) -> dict:
    """④ 成分变动记账：三种缺价政策的净值区间。

    ★ 成分变动 ≠ 退市。实测两个错法：
        无损移除（drop）  免费扔掉亏损 ⇒ 虚高约 +9pp
        全额清算（zero）  虚假暴跌     ⇒ 虚低约 −37pp
    正确做法只在权威退市日挂损失，工具无从得知，所以报区间。
    """
    out = {}
    for pol in ("drop", "hold_last", "zero"):
        pr = period_returns(wm, pm, policy=pol)
        out[pol] = float((1.0 + pr["ret"]).prod()) - 1.0
    rep.stats["policy_total"] = out

    pr0 = period_returns(wm, pm)
    n_ev = int((pr0["n_missing"] > 0).sum())
    w_miss = float(pr0["w_missing"].max())
    rep.stats["n_missing_events"] = n_ev

    if n_ev == 0:
        rep.add(OK, "成分变动记账",
                "没有任何一期出现「持有但无价格」的标的 —— 记账政策不影响结论",
                section=SECTION)
        return out

    # ★ 必须区分【停牌】与【退市】，面板自己就知道答案：区间末缺价、
    # 但之后又出现价格 ⇒ 那是停牌复牌，根本没有退市记账问题。
    # 真实 A 股面板实测（800 只 / 126 期）：24 个缺价标的-期事件
    # **100% 都是停牌**，无一永久消失。第一版把它们全报成
    # 「记账政策决定结论」并给 BLOCK —— 客户会去修一个不存在的退市问题，
    # 然后对后续所有告警都不再相信（factor-audit 把复牌报成复权错误的同一个坑）。
    n_resume, n_perm = _classify_missing(wm, pm)
    rep.stats["n_missing_resume"] = n_resume
    rep.stats["n_missing_permanent"] = n_perm

    lo, hi = min(out.values()), max(out.values())
    spread = hi - lo
    detail = (f"{n_ev} 期出现「持有但无价格」（单期最多占权重 {w_miss:.1%}）。"
              f"三种记账政策的累计收益：\n"
              f"无损移除 {out['drop']:+.2%} / 持有最后价 {out['hold_last']:+.2%}"
              f" / 全额清算 {out['zero']:+.2%}")

    if n_perm == 0 and n_resume > 0:
        # 全是停牌 ⇒ 这不是退市记账问题，但仍要说清停牌本身的影响
        rep.add(WARN, "缺价全是停牌，不是退市",
                detail + f"\n{n_resume} 个缺价标的-期事件【全部】在之后恢复交易，"
                f"无一永久消失 ⇒ 这不是退市记账问题",
                "所以上面的政策跨度【不适用】——「全额清算」在停牌上是错的记账"
                "（停牌股会复牌，不该记 −100%）。真正的问题是停牌期间的持仓"
                "无法调整也无法成交：请确认回测在停牌日没有假设可交易，"
                "并按复牌价而非停牌前价格结算",
                section=SECTION)
        return out

    if n_resume > 0:
        detail += (f"\n其中 {n_resume} 个标的-期事件之后恢复交易（停牌），"
                   f"仅 {n_perm} 个永久消失（疑似退市）")

    lvl = (BLOCK if spread > POLICY_SPREAD_BLOCK else
           WARN if spread > POLICY_SPREAD_WARN else OK)
    if lvl is OK:
        rep.add(OK, "成分变动记账", detail + f"\n跨度仅 {spread:.2%}",
                section=SECTION)
        return out

    # ★ 这里【不】声称哪种政策偏高。
    # 「无损移除必然最高」只在退市前有崩盘时成立（那时它扔掉的是真实亏损）；
    # 标的在中性价位消失时，实测 6 个面板里 3 个出现 drop < hold_last。
    # 报一个错的定向结论，比只报区间更糟。
    rep.add(lvl, "记账政策决定结论",
            detail + f"\n跨度 {spread:.2%} —— 换个政策就换个结论",
            "成分变动不是退市：无损移除会把退市前的亏损直接扔掉，"
            "全额清算会造出并未发生的暴跌。哪种偏高取决于退市前有没有崩盘，"
            "所以上面给的是区间而不是「正确答案」。"
            "损失只应挂在【权威退市日】—— 请补一列权威退市/停牌状态，"
            "在那之前这个跨度就是这份净值的记账不确定性下界",
            section=SECTION)
    return out
