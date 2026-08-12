"""族一：换手与成本。

★ 关键测试是【已知答案反推】：植入一个确切的费率，看毛净对账能不能
把它精确算回来。反推不出来说明成本口径与换手口径不一致 ——
而那种错误在净值上完全看不出来。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from strategy_audit import core, turnover_cost as tc
from strategy_audit.report import BLOCK, OK, WARN, AuditReport

from synth import equal_weight, make_prices, month_ends


def _setup(px, wm):
    rep = AuditReport()
    to = tc.check_turnover_basis(wm, px, rep)
    pr = core.period_returns(wm, px)
    ppy = core.periods_per_year(wm.index)
    return rep, to, pr["ret"], ppy


# ---------------- ① 换手口径 ----------------

def test_turnover_basis_reports_both(px, wm_clean):
    rep, to, _, _ = _setup(px, wm_clean)
    assert "turnover_naive" in rep.stats
    assert "turnover_drift_adj" in rep.stats
    assert np.isfinite(rep.stats["turnover_drift_adj"])


def test_naive_overstates_when_weights_static(px):
    """目标权重恒定 ⇒ 朴素口径报 0 换手，漂移口径报正换手。

    ★ 这是两个口径差别最干净的体现，且方向与直觉相反：
    「什么都没改」不等于「不用交易」。
    """
    reb = month_ends(px)
    codes = list(px.columns[:10])
    wm = pd.DataFrame(0.1, index=pd.Index(reb, name="date"), columns=codes)
    rep = AuditReport()
    to = tc.check_turnover_basis(wm, px[codes], rep)
    assert float(to["naive"].abs().max()) < 1e-15
    assert float(to["drift_adj"].mean()) > 0
    f = [f for f in rep.findings if "换手口径" in f.name][0]
    assert f.level == WARN and "低估" in f.name + f.detail


# ---------------- ② 反推换手 ----------------

def test_implied_turnover_reports_measured_ratio(px, wm_clean):
    """反推 vs 实测必须报出【实测比值】，不预设方向。

    我原以为反推一律低估（曾用全池信号自相关实测低估 2.6 倍），
    但用持仓权重自相关反推时，等权少量持仓上反而高估。
    所以工具报比值，不报「一定低估」。
    """
    rep, to, _, _ = _setup(px, wm_clean)
    rep2 = AuditReport()
    tc.check_implied_turnover(wm_clean, to, rep2)
    assert "turnover_implied_ratio" in rep2.stats
    r = rep2.stats["turnover_implied_ratio"]
    assert np.isfinite(r) and r > 0


def test_implied_turnover_skips_when_too_few_names(px):
    """持仓过少 ⇒ 自相关无法估计，必须显式记录跳过。"""
    reb = month_ends(px)[:6]
    wm = pd.DataFrame(0.0, index=pd.Index(reb, name="date"),
                      columns=list(px.columns[:2]))
    wm.iloc[:, 0] = 1.0
    rep = AuditReport()
    to = core.turnover(wm, px)
    tc.check_implied_turnover(wm, to, rep)
    assert rep.skipped and "反推换手" in rep.skipped[0]


# ---------------- ③ 盈亏平衡成本 ----------------

def test_breakeven_recovers_the_cost_that_kills_alpha(px, wm_clean):
    """在盈亏平衡点扣费 ⇒ 年化超额必须≈0（反推自洽）。"""
    rep, to, rets, ppy = _setup(px, wm_clean)
    res = tc.breakeven_cost(rets, to, ppy)
    be = res["be_bp"]
    assert np.isfinite(be) and be > 0
    t = to["drift_adj"].reindex(rets.index).fillna(0.0)
    net = rets - 2.0 * (be * 1e-4) * t
    ann = core.annualize(net, ppy)["ann_ret"]
    assert abs(ann) < 1e-4, f"在盈亏平衡成本处年化应为 0，实得 {ann}"


def test_breakeven_zero_when_gross_already_negative(px):
    """毛收益就为负 ⇒ BLOCK，且不该给出正的盈亏平衡成本。"""
    px_down = make_prices(n_codes=30, mu=-0.0015, seed=4)
    reb = month_ends(px_down)
    wm = equal_weight(px_down, reb, k=8, seed=5)
    rep, to, rets, ppy = _setup(px_down, wm)
    rep2 = AuditReport()
    tc.check_breakeven(rets, to, ppy, rep2)
    assert rep2.blockers
    assert "零成本" in rep2.blockers[0].name


def test_breakeven_uses_benchmark_as_excess(px, wm_clean):
    """给了基准 ⇒ 盈亏平衡按超额算，必须比绝对口径更严。"""
    rep, to, rets, ppy = _setup(px, wm_clean)
    abs_be = tc.breakeven_cost(rets, to, ppy)["be_bp"]
    bench = pd.Series(rets.values * 0.5, index=rets.index)
    exc_be = tc.breakeven_cost(rets, to, ppy, bench)["be_bp"]
    assert exc_be < abs_be


# ---------------- ④ 毛净对账 ----------------

def test_gross_net_reconcile_recovers_planted_fee(px, wm_clean):
    """★ 植入 15bp 单边费率 ⇒ 对账必须精确反推 15bp。"""
    rep, to, rets, ppy = _setup(px, wm_clean)
    t = to["drift_adj"].reindex(rets.index).fillna(0.0)
    net = rets - 2.0 * 15e-4 * t
    rep2 = AuditReport()
    tc.check_gross_net_reconcile(rets, net, to, rep2)
    assert abs(rep2.stats["implied_cost_bp"] - 15.0) < 1e-6
    assert not rep2.blockers and not rep2.warnings


def test_gross_net_reconcile_flags_uncharged_net(px, wm_clean):
    """声称是净值但根本没扣费 ⇒ BLOCK。"""
    rep, to, rets, ppy = _setup(px, wm_clean)
    rep2 = AuditReport()
    tc.check_gross_net_reconcile(rets, rets.copy(), to, rep2)
    assert rep2.blockers and "没扣成本" in rep2.blockers[0].name


def test_gross_net_reconcile_flags_unstable_basis(px, wm_clean):
    """净值里混了与换手无关的扣项 ⇒ 隐含费率不再是常数 ⇒ WARN。"""
    rep, to, rets, ppy = _setup(px, wm_clean)
    t = to["drift_adj"].reindex(rets.index).fillna(0.0)
    # 15bp 换手费 + 每期固定 20bp 管理费（与换手无关）
    net = rets - 2.0 * 15e-4 * t - 0.0020
    rep2 = AuditReport()
    tc.check_gross_net_reconcile(rets, net, to, rep2)
    assert any("口径不稳" in f.name for f in rep2.warnings)


def test_gross_net_skips_on_no_overlap(px, wm_clean):
    rep, to, rets, ppy = _setup(px, wm_clean)
    rep2 = AuditReport()
    tc.check_gross_net_reconcile(
        rets, pd.Series(dtype=float), to, rep2)
    assert rep2.skipped
