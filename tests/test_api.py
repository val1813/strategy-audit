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
    # ★ 新行为：认不出来的表不再让整份审计失败 —— 报 WARN 说清楚认不出，
    # 然后用【认得出的部分】继续审（这里价格表还在，族三仍能跑）。
    # 旧行为是直接 BLOCK 返回，等于因为一张表格式不对就什么都不给用户。
    idf = [f for f in rep.findings if f.name == "输入识别结果"][0]
    assert idf.level == WARN
    assert "认不出来" in idf.detail
    # 权重相关的检查必须出现在「未能检查」里，不能静默消失
    assert any("换手口径" in s for s in rep.skipped)
    assert any("权重前视" in s for s in rep.skipped)
    # 能力矩阵要如实反映「没有权重面板」
    assert "weights" not in rep.stats["capability"]


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


# ---------------- 傻瓜式：给什么审什么 ----------------

def test_nav_only_audits_significance_family():
    """★ 只有一条净值曲线也要能审 4 项，不能什么都不给。"""
    from strategy_audit import audit
    idx = pd.date_range("2016-01-31", periods=120, freq="ME")
    r = pd.Series(np.random.default_rng(11).normal(0.006, 0.045, 120), index=idx)
    df = pd.DataFrame({"日期": idx, "累计净值": (1 + r).cumprod().values})
    rep = audit(df)
    assert "nav" in rep.stats["capability"]
    assert "策略层显著性" in rep.sections()
    # 权重相关的 8 项必须列在「未能检查」
    assert len(rep.skipped) >= 8


def test_argument_order_does_not_matter(px, wm_clean):
    """★ 顺序无关：两个表调换位置结论必须一致。"""
    from strategy_audit import audit
    w, p = _long(px, wm_clean)
    a = audit(w, p)
    b = audit(p, w)
    key = lambda r: sorted((f.level, f.name) for f in r.findings
                           if f.section != "输入识别")
    assert key(a) == key(b)


def test_wide_tables_give_same_result_as_long(px, wm_clean):
    """宽表与长表必须等价。"""
    from strategy_audit import audit
    w, p = _long(px, wm_clean)
    a = audit(w, p)
    b = audit(wm_clean, px)
    key = lambda r: sorted((f.level, f.name) for f in r.findings
                           if f.section != "输入识别")
    assert key(a) == key(b)


def test_detection_result_always_reported(px, wm_clean):
    """★ 识别结果必须出现在报告里 —— 认错了用户得能看出来。"""
    from strategy_audit import audit
    w, p = _long(px, wm_clean)
    rep = audit(w, p)
    f = [f for f in rep.findings if f.name == "输入识别结果"][0]
    assert "权重面板" in f.detail and "价格面板" in f.detail
    assert "请核对" in f.impact


def test_no_input_blocks_with_guidance():
    from strategy_audit import audit
    rep = audit()
    assert rep.blockers
    assert "净值曲线" in rep.blockers[0].impact


def test_unknown_input_does_not_kill_the_audit(px, wm_clean):
    """认不出的表只报 WARN，能审的部分继续审。"""
    from strategy_audit import audit
    w, p = _long(px, wm_clean)
    junk = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    rep = audit(w, p, junk)
    idf = [f for f in rep.findings if f.name == "输入识别结果"][0]
    assert idf.level == WARN
    assert "认不出来" in idf.detail
    # 主体检查照跑
    assert any(f.section == "换手与成本" for f in rep.findings)


def test_net_must_be_explicit_keyword(px, wm_clean):
    """★ net 必须显式传 —— 收益率序列从数值上无法区分角色。"""
    from strategy_audit import audit, core
    w, p = _long(px, wm_clean)
    wmn = wm_clean.div(wm_clean.abs().sum(axis=1), axis=0).fillna(0.0)
    pr = core.period_returns(wmn, px)
    to = core.turnover(wmn, px)
    t = to["drift_adj"].reindex(pr.index).fillna(0.0)
    net = pr["ret"] - 2.0 * 14e-4 * t
    rep = audit(w, p, net=net)
    assert abs(rep.stats["implied_cost_bp"] - 14.0) < 0.5


def test_capability_matrix_in_report_text(px, wm_clean):
    from strategy_audit import audit
    w, p = _long(px, wm_clean)
    txt = audit(w, p).text()
    assert "能审" in txt and "审不了" in txt
    assert txt.index("能审") < txt.index("【输入识别】")
