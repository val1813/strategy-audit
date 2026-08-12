"""族一：换手与成本真实性。

审的是「这份净值的成本扣得够不够」，四项：

    ① 换手口径      朴素 0.5·Σ|Δw| vs 漂移调整口径，报差异方向与量级
    ② 反推 vs 实测  用持仓自相关反推换手，报它与实测换手的比值
    ③ 盈亏平衡成本  多大的单边成本会把年化超额吃到 0
    ④ 毛净对账      若同时给了毛净两条曲线，反推你【实际】计了多少 bp

★ 为什么这一族排第一
------------------
成本是唯一「不需要新数据、纯靠算对就能推翻结论」的缺陷。
实测过的案例：静态口径说筛掉最贵 10% 能省 2bp，组合层 41 个月实测
只有 +0.112bp —— 差 18 倍，结论从「值得做」翻成「噪声」。

★ 本模块不替客户选成本假设
------------------------
成本是客户的执行现实（券商费率、冲击、可成交比例），工具无从得知。
所以主输出是【盈亏平衡成本】—— 一个不需要假设的量：
「你的策略需要单边成本低于 X bp 才有正超额」。X 很小就是坏消息，
不管客户的真实成本是多少。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .core import annualize, periods_per_year, rank_autocorr_turnover, turnover
from .report import BLOCK, OK, WARN, AuditReport

SECTION = "换手与成本"

# 漂移调整与朴素口径相对差异超过这个比例 ⇒ 值得报出来
DRIFT_REL_TOL = 0.05

# 盈亏平衡单边成本门槛（bp）。A 股实测参考：
#   佣金+过户费+印花税(卖出千一) 单边约 8~12bp（机构费率）
#   加冲击成本，中小盘单边 20~30bp 是常见的现实值
# 所以盈亏平衡 < 10bp 基本等于「执行不了」。
BE_BLOCK_BP = 10.0
BE_WARN_BP = 30.0

# 毛净反推的隐含费率，10~90 分位相对跨度上限。
# ★ 固定费率下这个跨度恒为 0，所以 10% 已经很宽容。
# 用相对量而非绝对 bp：绝对门槛不随费率尺度缩放（实测放过了 20bp 管理费）。
IMPLIED_REL_SPREAD_TOL = 0.10


def _fmt_bp(x: float) -> str:
    return "∞" if not np.isfinite(x) else f"{x:.1f}bp"


def check_turnover_basis(wm: pd.DataFrame, pm: pd.DataFrame,
                         rep: AuditReport) -> pd.DataFrame:
    """① 换手口径：朴素 vs 漂移调整。"""
    to = turnover(wm, pm)
    naive = float(to["naive"].mean())
    adj = float(to["drift_adj"].mean())
    rep.stats["turnover_naive"] = naive
    rep.stats["turnover_drift_adj"] = adj

    # ★ 只有 adj 算不出来才该跳过。第一版还把 naive<=0 也跳了 ——
    # 那恰恰是最有信息量的情形：目标权重恒定（naive=0）时，
    # 价格漂移仍然逼你交易（adj>0）。跳过它等于把最干净的证据丢掉。
    if not np.isfinite(adj):
        rep.skip("换手口径对比", "价格缺失，无法计算漂移调整换手")
        return to
    if adj <= 0 and naive <= 0:
        rep.add(OK, "换手口径", "两个口径下换手均为 0（持仓从未变动且价格未漂移）",
                section=SECTION)
        return to

    rel = (naive - adj) / adj if adj > 0 else np.inf
    ppy = periods_per_year(wm.index)
    # 每期换手差 × 每年期数 × 2（双边）× 20bp 作为量级示意
    gap_ann_bp = abs(naive - adj) * ppy * 2 * 20 * 1e-4 * 1e4

    if abs(rel) < DRIFT_REL_TOL:
        rep.add(OK, "换手口径",
                f"单边换手 {adj:.1%}/期（漂移调整口径），"
                f"朴素口径 {naive:.1%}，相差 {rel:+.1%}（<{DRIFT_REL_TOL:.0%}）",
                section=SECTION)
        return to

    who = "高估" if rel > 0 else "低估"
    if naive <= 0:
        # 目标权重恒定的特例：朴素口径报 0，但你其实必须交易才能维持目标
        rep.add(WARN, "换手口径低估（目标权重恒定）",
                f"朴素 0.5·Σ|Δw| = 0.0%/期（目标权重从未改变），"
                f"漂移调整 = {adj:.1%}/期",
                "「什么都没改」不等于「不用交易」：价格漂移会让实际持仓"
                "偏离目标，拉回来就是一笔真实交易。"
                f"按双边 20bp 算，这段被漏掉的成本约值年化 {gap_ann_bp:.0f}bp",
                section=SECTION)
        return to

    rep.add(WARN, f"换手口径{who} {abs(rel):.0%}",
            f"朴素 0.5·Σ|Δw| = {naive:.1%}/期，"
            f"漂移调整 = {adj:.1%}/期，朴素口径{who} {abs(rel):.1%}",
            f"朴素口径把「价格涨了导致权重变大」也算成一笔交易，"
            f"而那笔交易并不存在（你没下单）。按双边 20bp 算，"
            f"这段差异约值年化 {gap_ann_bp:.0f}bp。"
            f"成本应按漂移调整口径计",
            section=SECTION)
    return to


def check_implied_turnover(wm: pd.DataFrame, to: pd.DataFrame,
                           rep: AuditReport) -> None:
    """② 由持仓自相关反推的换手 vs 实测换手。

    ★ 这一项只报【测出来的比值和方向】，不预设方向。
    我原本以为反推一律低估（曾实测低估 2.6 倍），但那是用
    「全池信号自相关」反推的；用「持仓权重自相关」反推时，
    在等权少量持仓的组合上反而会高估。方向取决于反推用的是哪个量，
    所以工具报实测比值，让客户看到自己这份数据上的偏差。
    """
    implied = rank_autocorr_turnover(wm)
    actual = float(to["drift_adj"].mean())
    if not np.isfinite(implied) or not np.isfinite(actual) or actual <= 0:
        rep.skip("反推换手对比", "持仓过少或换手为 0，自相关无法估计")
        return

    ratio = implied / actual
    rep.stats["turnover_implied"] = implied
    rep.stats["turnover_implied_ratio"] = ratio

    if 0.8 <= ratio <= 1.25:
        rep.add(OK, "反推换手",
                f"由持仓自相关反推 {implied:.1%}/期，实测 {actual:.1%}/期"
                f"（比值 {ratio:.2f}）—— 这份数据上反推可用",
                section=SECTION)
        return

    who = "高估" if ratio > 1 else "低估"
    rep.add(WARN, f"反推换手{who} {abs(ratio - 1):.0%}",
            f"由持仓自相关反推 {implied:.1%}/期，按权重实测 {actual:.1%}/期"
            f"（比值 {ratio:.2f}）",
            "若你的成本是按自相关反推的换手计的，成本口径就错了这个倍数。"
            "换手必须按实际权重变化算，不能从自相关近似 —— "
            "偏差方向随组合结构（持仓数、是否等权）而变，不能靠经验修正",
            section=SECTION)


def breakeven_cost(rets: pd.Series, to: pd.DataFrame,
                   ppy: float, bench: pd.Series | None = None) -> dict:
    """③ 盈亏平衡单边成本（bp）。

    解 c 使得   年化(毛收益 − 2·c·换手) − 年化(基准) = 0
    双边成本按 2·c·单边换手 计。
    """
    r = pd.Series(rets).astype(float)
    t = to["drift_adj"].reindex(r.index)
    if t.isna().all():
        t = to["naive"].reindex(r.index)
    t = t.fillna(0.0)

    base = 0.0
    if bench is not None:
        b = pd.Series(bench).reindex(r.index).astype(float)
        base = annualize(b.dropna(), ppy)["ann_ret"] or 0.0
        if not np.isfinite(base):
            base = 0.0

    def excess(c_bp: float) -> float:
        net = r - 2.0 * (c_bp * 1e-4) * t
        a = annualize(net, ppy)["ann_ret"]
        return (a if np.isfinite(a) else -1.0) - base

    if excess(0.0) <= 0:
        return dict(be_bp=0.0, gross_ann=annualize(r, ppy)["ann_ret"],
                    bench_ann=base, already_negative=True)

    lo, hi = 0.0, 1.0
    while excess(hi) > 0 and hi < 5000.0:
        hi *= 2.0
    if excess(hi) > 0:
        return dict(be_bp=np.inf, gross_ann=annualize(r, ppy)["ann_ret"],
                    bench_ann=base, already_negative=False)
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if excess(mid) > 0:
            lo = mid
        else:
            hi = mid
    return dict(be_bp=0.5 * (lo + hi), gross_ann=annualize(r, ppy)["ann_ret"],
                bench_ann=base, already_negative=False)


def check_breakeven(rets: pd.Series, to: pd.DataFrame, ppy: float,
                    rep: AuditReport, bench: pd.Series | None = None) -> None:
    """③ 报盈亏平衡成本。"""
    res = breakeven_cost(rets, to, ppy, bench)
    be = res["be_bp"]
    rep.stats["breakeven_bp"] = be
    rep.stats["gross_ann_ret"] = res["gross_ann"]

    ann = res["gross_ann"]
    ann_s = f"{ann:.2%}" if np.isfinite(ann) else "n/a"
    vs = "（对基准超额）" if bench is not None else "（绝对，rf=0）"

    if res.get("already_negative"):
        rep.add(BLOCK, "零成本下已无正超额",
                f"毛年化 {ann_s}{vs}，还没扣任何成本就不为正",
                "成本只会让它更差。这份净值不支持「策略有效」的结论",
                section=SECTION)
        return

    lvl = (BLOCK if be < BE_BLOCK_BP else
           WARN if be < BE_WARN_BP else OK)
    t_ann = float(to["drift_adj"].reindex(pd.Series(rets).index)
                  .fillna(to["naive"].mean()).mean()) * ppy
    detail = (f"单边成本需低于 {_fmt_bp(be)} 才有正超额"
              f"（毛年化 {ann_s}{vs}，年化单边换手 {t_ann:.1f}x）")
    if lvl is OK:
        rep.add(OK, "盈亏平衡成本", detail, section=SECTION)
    else:
        rep.add(lvl, "盈亏平衡成本偏低", detail,
                f"A 股机构单边佣金过户印花约 8~12bp，加冲击后中小盘常见 "
                f"20~30bp。盈亏平衡 {_fmt_bp(be)} 意味着"
                f"{'执行不了' if lvl is BLOCK else '只在最优执行下勉强成立'} —— "
                f"这不需要你告诉我真实费率，是这份换手自己的算术结果",
                section=SECTION)


def check_gross_net_reconcile(gross: pd.Series, net: pd.Series,
                              to: pd.DataFrame, rep: AuditReport) -> None:
    """④ 毛净对账：反推你【实际】扣了多少 bp。

    ★ 这一项抓的是「声称扣了成本，其实没扣干净」。
    实测教训：闸门自己也会标错，所以用对账型断言 —— 由两条曲线
    之差反推隐含费率，再跟客户声称的口径对比。
    """
    g = pd.Series(gross).astype(float)
    n = pd.Series(net).astype(float)
    idx = g.index.intersection(n.index)
    if len(idx) < 3:
        rep.skip("毛净对账", "毛/净收益序列重叠不足 3 期")
        return

    diff = (g.loc[idx] - n.loc[idx])
    t = to["drift_adj"].reindex(idx)
    if t.isna().all():
        t = to["naive"].reindex(idx)
    t = t.reindex(idx)
    ok = t.notna() & (t > 0) & diff.notna()
    if ok.sum() < 3:
        rep.skip("毛净对账", "有效重叠期不足（换手为 0 或缺失）")
        return

    # diff ≈ 2·c·turnover  ⇒  c = diff / (2·turnover)
    implied_bp = (diff[ok] / (2.0 * t[ok])) * 1e4
    med = float(implied_bp.median())
    lo, hi = float(implied_bp.quantile(0.1)), float(implied_bp.quantile(0.9))
    rep.stats["implied_cost_bp"] = med

    if med < 0.5:
        rep.add(BLOCK, "净收益几乎没扣成本",
                f"毛净之差反推隐含单边成本中位数仅 {med:.2f}bp"
                f"（10~90 分位 {lo:.2f}~{hi:.2f}bp）",
                "这条「净值」实际上是毛净值。若报告里写了「已扣交易成本」，"
                "那句话与数据不符",
                section=SECTION)
        return

    # ★ 判「是否固定费率」必须用【相对】跨度，不能用绝对 bp。
    # 第一版门槛 max(5bp, 0.5·中位数) 在中位数 27bp 时放宽到 13.7bp，
    # 于是一笔每期 20bp 的固定管理费（相对跨度 26%）被判成 OK。
    # 真正的固定费率下相对跨度恒为 0 —— 所以 10% 已经很宽容了。
    spread = hi - lo
    rel_spread = spread / abs(med) if abs(med) > 1e-9 else np.inf
    rep.stats["implied_cost_rel_spread"] = rel_spread

    if rel_spread > IMPLIED_REL_SPREAD_TOL:
        rep.add(WARN, "隐含成本口径不稳",
                f"反推隐含单边成本中位数 {med:.1f}bp，"
                f"但 10~90 分位跨 {lo:.1f}~{hi:.1f}bp"
                f"（相对跨度 {rel_spread:.0%}）",
                "固定费率下这个反推值应当【恒为常数】（相对跨度 0）。"
                "跨度这么大说明净收益里还混了与换手无关的扣项"
                "（管理费、融券费、现金拖累），或冲击成本模型是非线性的，"
                "或换手口径与扣费口径不一致 —— "
                "此时「已扣成本 X bp」这句话无法对账",
                section=SECTION)
        return

    rep.add(OK, "毛净对账",
            f"反推隐含单边成本 {med:.1f}bp（10~90 分位 {lo:.1f}~{hi:.1f}bp，"
            f"相对跨度 {rel_spread:.1%}），与固定费率一致", section=SECTION)
