"""族六：处方层 —— 唯一会说「怎么改」的一族。

前五族只回答「这份净值可不可信」。这一族回答「按这份持仓，
有没有一个【不需要任何预测】的改动能确定地改善净收益」。

三项：

    ① 换手的边际价值   跟「什么都不做」比，这次调仓在毛口径上买到了什么
    ② 换手成分拆分     换手里多少是名单换血、多少是权重微调（决定旋钮存不存在）
    ③ 身份受控前沿     削权重微调的交叉成本 c*(φ)，无需成本假设

★ 为什么处方层必须自带闸门（这一族存在的全部理由）
------------------------------------------------
一个不带闸门的优化建议器就是一台过拟合机器。实测（真实 A 股面板
800 只 / 2016-2026 / 月频 30 只）：扫描无交易带宽 b，最优 b=12%
把换手从 96% 砍到 25%、净年化 −0.60% → +2.96%，全样本 +356bp ——
一条看起来极漂亮的建议。但在【随机选股】组合上调同一个旋钮，
改善是 +218 ~ +705bp（均值 +371），本账的 +356bp 完整落在噪声里；
半样本样本外更是 −1518 ~ +932bp。

所以本族的默认输出是【不可处方】，只有过了四关才给建议。
过不了关时报告必须写明「这个旋钮在你的数据上不可调」并给最小可检出效应，
而不是给一个噪声建议 —— 后者比不给建议糟得多。

★ 身份必须钉住，否则「改善」是换了个组合
------------------------------------
第一版按 |Δw| 排序削预算（不分名单变动与权重微调），结果 φ=0.4 时
最大权重 95.7%、有效注数 1.5、与目标名单重合仅 1% —— 那个 +21.37%
的「毛收益改善」是一只股票，不是省下的成本。

这与族四漏传权重是同一个病（对照组多变了一个东西），修法也一样：

    名单变动  进场/出场          ⇒ 【必须全执行】，身份由此钉住
    权重微调  两边都持有的名字   ⇒ 受预算份额 φ 约束

于是名单重合度在构造上恒为 100%，τ(φ) 在构造上单调 —— 这两条不是
实证发现而是算术保证，测试里按恒等式断言。

★ 为什么只削权重微调是【无预测】的
--------------------------------
小额权重修补是唯一可以先验断定「收益期望≈0 而成本确定发生」的那类交易：
把 1.9% 拉回 2.0% 不表达任何观点。名单变动表达观点（这是策略本身），
所以不许动。Novy-Marx & Velikov (2016) 的 buy/hold sS 规则同族，
差别是它改变名单变动的数量，本族刻意不碰名单。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .core import _seg_gross, align, annualize, drift_weights
from .report import OK, WARN, AuditReport

SECTION = "处方（怎么改）"

# 预算份额网格。★ 必须【预先声明】且不许按结果增删 ——
# 「扫一个更细的网格找更好的点」正是过拟合本身。
PHI_GRID = (1.0, 0.8, 0.6, 0.4, 0.2, 0.0)

# 关A：权重微调占换手的份额低于此 ⇒ 旋钮不存在，削了也省不下换手。
# ★ 实测标定：月频 30 只反转组合微调仅占 1%（名单换血 95%）⇒ 不可处方；
# 市值前 200 只市值加权微调占 73% ⇒ 可处方。20% 落在两者之间且靠近下界，
# 因为门槛的作用是挡掉「省不下东西」的情形，不是挑出最好的情形。
TWEAK_SHARE_FLOOR = 0.20

# 关B：削微调后毛收益的逐期配对 t 低于此 ⇒ 毛收益显著变差，不许处方
GROSS_HARM_T = -2.0

# 至少要省下这么多单边换手才值得说（每期）
MIN_TAU_SAVED = 1e-4


def _norm_gross(wm: pd.DataFrame) -> pd.DataFrame:
    """把每期权重按 gross（绝对值和）归一到 1。

    ★ 本模块的每个入口都必须先过这一道，否则结论会随【口径】而变。
    实测（对账抓到的）：core.period_returns 不归一（收益随 gross 线性缩放），
    而本模块按单位资本算收益（与 core.daily_path 一致）。两者在 gross=1
    时恒等，但在下面三种输入上差一个 1/gross 的常数因子：

        市场中性 Σw=0、gross=2   ⇒ 差 0.5×
        留了现金 Σw=0.8          ⇒ 差 1.25×
        2 倍杠杆 Σw=2            ⇒ 差 0.5×

    audit() 上游已经归一过（contract.normalize_gross），所以走公开 API
    时看不到这个差；但直接调用本模块的函数就会看到 —— 而「交叉成本 c*」
    是个 bp 量，差一个 2 倍就是把结论说错。归一在入口做掉最省心。
    """
    gross = wm.abs().sum(axis=1).replace(0.0, np.nan)
    return wm.div(gross, axis=0).fillna(0.0)


def split_turnover(wm: pd.DataFrame, pm: pd.DataFrame) -> dict:
    """把换手拆成【名单换血】与【权重微调】两部分。

    名单换血  进场（原来没有、现在有）+ 出场（原来有、现在没有）
    权重微调  两边都持有的名字之间的权重差

    ★ 这个拆分决定了处方层有没有可动的旋钮，所以它排在所有处方之前。
    两部分之和恒等于漂移调整口径的总换手（测试按恒等式断言）。
    """
    wmA, pmA = align(_norm_gross(wm), pm)
    dates = list(wmA.index)
    churn, tweak = [], []
    for cur, nxt in zip(dates[:-1], dates[1:]):
        g = _seg_gross(pmA, cur, nxt)
        drift = drift_weights(wmA.loc[cur], g)
        tgt = wmA.loc[nxt]
        h = drift.abs() > 1e-12
        s = tgt.abs() > 1e-12
        dw = tgt - drift
        churn.append(0.5 * float(dw[(h & ~s) | (s & ~h)].abs().sum()))
        tweak.append(0.5 * float(dw[h & s].abs().sum()))
    if not churn:
        return dict(churn=np.nan, tweak=np.nan, share=np.nan, n=0)
    c, t = float(np.mean(churn)), float(np.mean(tweak))
    return dict(churn=c, tweak=t, share=t / max(c + t, 1e-12),
                n=len(churn))


def turnover_value(wm: pd.DataFrame, pm: pd.DataFrame) -> pd.DataFrame:
    """① 每次调仓在毛口径上买到了什么（vs 什么都不做）。

    对每个调仓日 t：
        w̃    上期权重被区间收益推着漂移 = 你【什么都不做】的持仓
        w    实际目标权重
        Δ    (w − w̃)·r_{t→t+1}   ← 这次调仓的毛口径边际价值

    ★ 零成本下 Δ ≤ 0 就已经说明换手是纯损耗 —— 这个结论不需要
    任何成本假设，也比盈亏平衡成本更强：盈亏平衡说「成本要低于 X bp」，
    这一项说「就算成本是 0 也不划算」。
    """
    wmA, pmA = align(_norm_gross(wm), pm)
    dates = list(wmA.index)
    rows = []
    for k in range(len(dates) - 2):
        prev, cur, nxt = dates[k], dates[k + 1], dates[k + 2]
        g = _seg_gross(pmA, prev, cur)
        w_d = drift_weights(wmA.loc[prev], g)
        w_a = wmA.loc[cur]
        r = _seg_gross(pmA, cur, nxt) - 1.0
        rv = r.where(np.isfinite(r), 0.0)
        dw = w_a - w_d
        tau = 0.5 * float(dw.abs().sum())
        if tau <= 1e-9:
            continue
        rows.append(dict(date=cur, tau=tau, delta=float((dw * rv).sum())))
    return pd.DataFrame(rows).set_index("date") if rows else pd.DataFrame()


def identity_controlled_path(wm: pd.DataFrame, pm: pd.DataFrame,
                             phi: float) -> dict:
    """按预算份额 φ 重放净值，【身份受控】。

    名单变动无条件全执行；两边都持有的名字按 |Δw| 从大到小执行，
    直到用掉 φ 份额的微调预算。

    返回 dict(ret, tau, overlap, maxw)。overlap 应恒为 1.0 ——
    若不是，构造有污染（测试按此断言）。
    """
    wmA, pmA = align(_norm_gross(wm), pm)
    dates = list(wmA.index)
    hold = wmA.loc[dates[0]].copy()
    rets, taus, ovs, mxs = [], [], [], []
    for cur, nxt in zip(dates[:-1], dates[1:]):
        g = _seg_gross(pmA, cur, nxt)
        r = g.where(np.isfinite(g), 1.0) - 1.0
        base = float(hold.abs().sum())
        rets.append(float((hold * r).sum() / base) if base > 0 else 0.0)

        drift = drift_weights(hold, g)
        tgt = wmA.loc[nxt]
        h = drift.abs() > 1e-12
        s = tgt.abs() > 1e-12

        # ① 名单变动：进场、出场 —— 无条件全执行（身份由此钉住）
        new = drift.copy()
        enter = s & ~h
        exit_ = h & ~s
        new[enter] = tgt[enter]
        new[exit_] = 0.0

        # ② 权重微调：按 |Δw| 从大到小执行到预算用尽
        both = h & s
        if both.any():
            if phi >= 1.0:
                new[both] = tgt[both]
            elif phi > 0.0:
                dw = (tgt - drift)[both]
                limit = phi * float(dw.abs().sum())
                used = 0.0
                for c, v in dw.abs().sort_values(ascending=False).items():
                    if used + v > limit:
                        break
                    new[c] = tgt[c]
                    used += v

        gross = float(new.abs().sum())
        new = new / gross if gross > 0 else new
        taus.append(0.5 * float((new - drift).abs().sum()))
        nz = new[new.abs() > 1e-12]
        tn = tgt[s].index
        ovs.append(len(set(nz.index) & set(tn)) / max(len(tn), 1))
        mxs.append(float(nz.abs().max()) if len(nz) else np.nan)
        hold = new
    idx = pd.Index(dates[1:], name="date")
    return dict(ret=pd.Series(rets, index=idx),
                tau=pd.Series(taus, index=idx),
                overlap=float(np.mean(ovs)) if ovs else np.nan,
                maxw=float(np.nanmean(mxs)) if mxs else np.nan)


def crossover_cost(wm: pd.DataFrame, pm: pd.DataFrame, ppy: float,
                   grid=PHI_GRID) -> pd.DataFrame:
    """③ 身份受控的换手前沿与交叉成本 c*(φ)。

        c*(φ) = [G(1) − G(φ)] / [2·(τ(1) − τ(φ))·ppy]

    单边成本高于 c*(φ) 时，把权重微调削到 φ 的净收益严格更优。
    ★ 这个量不需要任何成本假设 —— 是两条毛曲线自己的算术。
    """
    rows = []
    base = None
    for p in grid:
        r = identity_controlled_path(wm, pm, p)
        a = annualize(r["ret"], ppy)
        rec = dict(phi=p, tau=float(r["tau"].mean()), gross=a["ann_ret"],
                   sharpe=a["sharpe"], overlap=r["overlap"], _ret=r["ret"])
        if base is None:
            base = rec
        rows.append(rec)

    g1, t1, r1 = base["gross"], base["tau"], base["_ret"]
    for rec in rows:
        saved = t1 - rec["tau"]
        rec["tau_saved"] = saved
        rec["cstar"] = ((g1 - rec["gross"]) / (2.0 * saved * ppy) * 1e4
                        if saved > MIN_TAU_SAVED else np.nan)
        # 毛收益的逐期配对差（关B）与后半样本（关C）
        d = (rec["_ret"] - r1).dropna()
        se = float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 2 else np.nan
        rec["gross_diff"] = float(d.mean()) if len(d) else np.nan
        rec["gross_t"] = (float(d.mean() / se)
                          if se and np.isfinite(se) and se > 0 else np.nan)
        rec["gross_se"] = se
        h = len(d) // 2
        rec["oos_diff"] = float(d.iloc[h:].mean()) if len(d) > 3 else np.nan
    out = pd.DataFrame([{k: v for k, v in r.items() if k != "_ret"}
                        for r in rows])
    return out.set_index("phi")


# ---------------- 报告层 ----------------

def check_turnover_value(wm: pd.DataFrame, pm: pd.DataFrame,
                         rep: AuditReport) -> pd.DataFrame:
    """① 换手的边际价值：跟什么都不做比，换手买到了什么。"""
    d = turnover_value(wm, pm)
    if len(d) < 6:
        rep.skip("换手的边际价值",
                 f"有效调仓期只有 {len(d)} 期（需要 ≥6）")
        return d

    you = float(d["delta"].mean())
    se = float(d["delta"].std(ddof=1) / np.sqrt(len(d)))
    t = you / se if se > 0 else np.nan
    tau = float(d["tau"].mean())
    mde = 1.96 * se
    rep.stats["turnover_value_bp"] = you * 1e4
    rep.stats["turnover_value_t"] = t
    rep.stats["turnover_value_mde_bp"] = mde * 1e4

    detail = (f"每期单边换手 {tau:.1%}，在【毛口径】上买到 "
              f"{you * 1e4:+.1f}bp/期（{len(d)} 期，t={t:+.2f}）"
              f"\n对照基准是「什么都不做」：上期权重被价格推着漂移的持仓"
              f"\n本样本最小可检出 {mde * 1e4:.1f}bp/期")

    if np.isfinite(t) and t <= -2.0:
        rep.add(WARN, "换手在毛口径上就是负贡献", detail,
                f"零成本下这些交易每期已经亏 {abs(you) * 1e4:.1f}bp —— "
                "成本只会让它更差。这比盈亏平衡成本更强：那一项说"
                "「成本要低于 X bp」，这一项说「就算成本是 0 也不划算」。"
                "请核对信号在调仓频率上是否还有预测力",
                section=SECTION)
        return d

    if you <= 0:
        rep.add(WARN, "换手的毛口径贡献为负但不显著", detail,
                f"每期 {you * 1e4:+.1f}bp，t={t:+.2f} 与噪声不可区分。"
                f"「没查出问题」和「查不出这么小的问题」是两回事 —— "
                f"本样本只能识别 {mde * 1e4:.1f}bp/期以上的效应",
                section=SECTION)
        return d

    rep.add(OK, "换手的边际价值",
            detail + " —— 毛口径为正，换手买到了东西", section=SECTION)
    return d


def check_turnover_split(wm: pd.DataFrame, pm: pd.DataFrame,
                         rep: AuditReport) -> dict:
    """② 换手成分拆分：旋钮存不存在。"""
    s = split_turnover(wm, pm)
    if not s["n"] or not np.isfinite(s["share"]):
        rep.skip("换手成分拆分", "调仓期数不足或换手为 0")
        return s

    rep.stats["turnover_churn"] = s["churn"]
    rep.stats["turnover_tweak"] = s["tweak"]
    rep.stats["turnover_tweak_share"] = s["share"]

    detail = (f"单边换手 {s['churn'] + s['tweak']:.2%}/期 拆成："
              f"名单换血 {s['churn']:.2%}（进出场）+ "
              f"权重微调 {s['tweak']:.2%}（两边都持有的名字）"
              f"\n微调占 {s['share']:.0%}")
    rep.add(OK, "换手成分拆分",
            detail + ("\n⇒ 微调占比足够，存在可削的预算（见下一项）"
                      if s["share"] >= TWEAK_SHARE_FLOOR else
                      "\n⇒ 换手主要来自名单换血，那是策略本身的观点，不可削"),
            section=SECTION)
    return s


def check_prescription(wm: pd.DataFrame, pm: pd.DataFrame, ppy: float,
                       rep: AuditReport) -> pd.DataFrame | None:
    """③ 身份受控前沿 + 四道闸门，只在全过时给处方。

    ★ 默认输出是【不可处方】。这一项的价值一半在于它会拒绝。
    """
    s = split_turnover(wm, pm)
    if not s["n"] or not np.isfinite(s["share"]):
        rep.skip("削换手处方", "调仓期数不足或换手为 0")
        return None

    # ---- 关A：旋钮存不存在 ----
    if s["share"] < TWEAK_SHARE_FLOOR:
        rep.add(OK, "不可处方：没有可削的换手预算",
                f"权重微调只占换手的 {s['share']:.0%}"
                f"（门槛 {TWEAK_SHARE_FLOOR:.0%}）—— "
                f"换手的 {1 - s['share']:.0%} 是名单换血",
                "名单变动表达的是策略本身的观点，削它就是改策略，"
                "本工具不建议。要降换手只能降低调仓频率或收窄选股范围，"
                "那会改变策略身份 —— 那是你的决定，不是审计能替你做的。"
                "★ 本项【拒绝给出建议】，不是没查出问题",
                section=SECTION)
        return None

    d = crossover_cost(wm, pm, ppy)
    if len(d) < 2:
        rep.skip("削换手处方", "前沿无法计算")
        return None

    # ★ 身份受控的算术保证：名单重合恒为 100%、τ 单调。
    # 若断言不成立就是实现有 bug，绝不能带着污染的结论继续。
    ov = float(np.nanmin(d["overlap"]))
    taus = d["tau"].values
    if ov < 0.999 or not np.all(np.diff(taus) <= 1e-9):
        rep.skip("削换手处方",
                 f"身份受控构造自检未通过（名单重合 {ov:.1%}、"
                 f"τ {'单调' if np.all(np.diff(taus) <= 1e-9) else '非单调'}）"
                 "—— 不出结论，避免给出被污染的建议")
        return d

    cand = d.iloc[1:].copy()
    cand = cand[cand["tau_saved"] > MIN_TAU_SAVED]
    # ---- 关B：毛收益不能显著变差；关C：后半样本不能反转 ----
    ok = cand[(cand["gross_t"] > GROSS_HARM_T)
              & ((cand["oos_diff"] >= 0)
                 | (cand["oos_diff"].abs() < 1.96 * cand["gross_se"]))]

    lines = [f"权重微调占换手 {s['share']:.0%} ⇒ 旋钮存在。"
             f"削预算到 φ（名单变动仍全执行，名单重合恒 100%）："]
    for phi, row in d.iterrows():
        if phi >= 1.0:
            lines.append(f"  φ=1.0（当前）换手 {row['tau']:.2%}/期、"
                         f"毛年化 {row['gross']:+.2%}")
            continue
        cs = ("n/a" if not np.isfinite(row["cstar"])
              else f"{row['cstar']:.1f}bp")
        lines.append(f"  φ={phi:.1f} 换手 {row['tau']:.2%}/期、"
                     f"毛年化 {row['gross']:+.2%}、"
                     f"毛差 {row['gross_diff'] * 1e4:+.1f}bp/期"
                     f"（t={row['gross_t']:+.2f}）、交叉成本 {cs}")
    detail = "\n".join(lines)

    if ok.empty:
        rep.add(OK, "不可处方：削微调会显著损失毛收益", detail,
                "旋钮存在，但每一档 φ 都过不了「毛收益不显著变差」"
                "或「后半样本仍成立」这两关。"
                "★ 本项【拒绝给出建议】—— 一个过不了闸门的优化建议"
                "比不给建议糟得多",
                section=SECTION)
        return d

    # 取省下换手最多的那一档（不是 c* 最优的那一档 ——
    # ★ 按 c* 挑就是在噪声里挑最小值，那正是探针④的死法）
    pick = ok["tau_saved"].idxmax()
    row = ok.loc[pick]
    saved_rel = row["tau_saved"] / d.loc[1.0, "tau"]
    cs = row["cstar"]
    rep.stats["prescribe_phi"] = float(pick)
    rep.stats["prescribe_tau_saved"] = float(row["tau_saved"])
    rep.stats["prescribe_cstar_bp"] = float(cs) if np.isfinite(cs) else np.nan

    impact = (f"把权重微调削到 φ={pick:.1f}：换手 "
              f"{d.loc[1.0, 'tau']:.2%} → {row['tau']:.2%}/期"
              f"（省 {saved_rel:.0%}），毛年化 "
              f"{d.loc[1.0, 'gross']:+.2%} → {row['gross']:+.2%}。")
    if np.isfinite(cs) and cs <= 0:
        impact += ("交叉成本为负 ⇒ 换手更低【且】毛收益更高，"
                   "在任何成本水平下都严格占优 —— 不需要你的费率。")
    else:
        impact += (f"单边成本高于 {cs:.1f}bp 时这个改动的净收益更优。")
    impact += ("\n★ 名单变动一笔没动，所以这不改变你的选股观点，"
               "只是不再为「把 1.9% 拉回 2.0%」付钱。"
               "\n★ 归因：这类结论在市值加权组合上是【结构性】的"
               "（任何同类组合都适用），不是你选股的功劳 —— "
               "汇报时不要算成 alpha")
    rep.add(WARN, f"可处方：削微调省 {saved_rel:.0%} 换手", detail, impact,
            section=SECTION)
    return d
