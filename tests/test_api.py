"""顶层 API 与报告层。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from strategy_audit import audit_strategy
from strategy_audit.report import BLOCK, OK, WARN, AuditReport

from synth import (equal_weight, lookahead_weight, make_prices, month_ends,
                   to_long)


def _long(px, wm):
    return to_long(wm, "weight"), to_long(px, "close")


def test_clean_strategy_has_no_blockers(px, wm_clean):
    w, p = _long(px, wm_clean)
    rep = audit_strategy(w, p, name="干净")
    assert not rep.blockers, [(f.name, f.detail) for f in rep.blockers]
    assert rep.trustworthy


def test_planted_lookahead_is_blocked(px, wm_dirty):
    w, p = _long(px, wm_dirty)
    rep = audit_strategy(w, p, name="植入前视")
    assert rep.blockers
    assert any("同期收益" in f.name for f in rep.blockers)
    assert not rep.trustworthy


def test_report_groups_by_section(px, wm_clean):
    w, p = _long(px, wm_clean)
    rep = audit_strategy(w, p)
    secs = rep.sections()
    assert "前视与记账" in secs and "换手与成本" in secs
    txt = rep.text()
    # 前视必须排在成本【之前】：先问净值是不是真的，再问扣够没有
    assert txt.index("前视与记账") < txt.index("换手与成本")


def test_report_lists_skipped_explicitly(px, wm_clean):
    """未提供 net_returns ⇒ 毛净对账必须出现在「未能检查」里。

    ★ 静默跳过 = 客户以为查过了。
    """
    w, p = _long(px, wm_clean)
    rep = audit_strategy(w, p)
    assert any("毛净对账" in s for s in rep.skipped)
    txt = rep.text()
    assert "未能检查" in txt and "没有查过" in txt


def test_net_returns_enables_reconcile(px, wm_clean):
    """给了净收益 ⇒ 毛净对账不再跳过，且能反推植入的费率。"""
    from strategy_audit import core
    w, p = _long(px, wm_clean)
    wmn = wm_clean.div(wm_clean.abs().sum(axis=1), axis=0).fillna(0.0)
    pr = core.period_returns(wmn, px)
    to = core.turnover(wmn, px)
    t = to["drift_adj"].reindex(pr.index).fillna(0.0)
    net = pr["ret"] - 2.0 * 12e-4 * t
    rep = audit_strategy(w, p, net_returns=net)
    assert not any("毛净对账" in s for s in rep.skipped)
    assert abs(rep.stats["implied_cost_bp"] - 12.0) < 0.5


def test_stats_populated(px, wm_clean):
    w, p = _long(px, wm_clean)
    rep = audit_strategy(w, p)
    for k in ("n_dates", "n_names", "span", "turnover_drift_adj",
              "breakeven_bp", "periods_per_year"):
        assert k in rep.stats, k


def test_too_few_rebalances_blocks(px):
    reb = month_ends(px)[:2]
    wm = equal_weight(px, reb, k=5)
    w, p = _long(px, wm)
    rep = audit_strategy(w, p)
    assert rep.blockers and "调仓期数不足" in rep.blockers[0].name


def test_contract_failure_returns_early(px, wm_clean):
    """契约 BLOCK ⇒ 不该继续跑后面的检查（结果无意义）。"""
    w, p = _long(px, wm_clean)
    w = w.drop(columns=["weight"])
    rep = audit_strategy(w, p)
    assert rep.blockers
    assert len(rep.findings) == 1, "契约失败后不该再跑其他检查"


def test_benchmark_tightens_breakeven(px, wm_clean):
    from strategy_audit import core
    w, p = _long(px, wm_clean)
    wmn = wm_clean.div(wm_clean.abs().sum(axis=1), axis=0).fillna(0.0)
    pr = core.period_returns(wmn, px)
    bench = pd.Series(pr["ret"].values * 0.5, index=pr.index)
    be_abs = audit_strategy(w, p).stats["breakeven_bp"]
    be_exc = audit_strategy(w, p, benchmark=bench).stats["breakeven_bp"]
    assert be_exc < be_abs


# ---------------- 报告层 ----------------

def test_report_text_orders_block_first():
    rep = AuditReport(title="T")
    rep.add(OK, "好", "d", section="S")
    rep.add(BLOCK, "坏", "d", section="S")
    rep.add(WARN, "中", "d", section="S")
    txt = rep.text()
    assert txt.index("坏") < txt.index("中") < txt.index("好")


def test_report_multiline_detail_and_impact_indented():
    rep = AuditReport()
    rep.add(WARN, "n", "第一行\n第二行", "影响一\n影响二", section="S")
    lines = rep.text().split("\n")
    assert any(ln.strip() == "第二行" for ln in lines)
    assert any("⇒ 影响一" in ln for ln in lines)
    # 续行不重复箭头
    assert sum("⇒" in ln for ln in lines) == 1


def test_trustworthy_only_about_blockers():
    rep = AuditReport()
    rep.add(WARN, "w", "d")
    assert rep.trustworthy
    rep.add(BLOCK, "b", "d")
    assert not rep.trustworthy
