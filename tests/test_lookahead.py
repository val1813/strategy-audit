"""族二：前视与记账。

★ 两类断言缺一不可
    植入缺陷必须被抓到   —— 否则检查没用
    干净组合必须不报警   —— 否则检查会被关掉

第二类更容易被忽略。第一版的调仓日对齐检查在【干净】组合上报 WARN
（相对多 20%，实为 1.4σ 噪声），因为门槛只看比例、没有噪声模型。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from strategy_audit import lookahead as la
from strategy_audit.report import BLOCK, OK, WARN, AuditReport

from synth import (equal_weight, lookahead_weight, make_prices, month_ends,
                   tilted_weight)


def _run(wm, px, fn):
    rep = AuditReport()
    fn(wm, px, rep)
    return rep


# ---------------- ① 调仓日对齐 ----------------

def test_day0_clean_portfolio_does_not_warn(px, wm_clean):
    """★ 干净组合不能报警（第一版在这里假阳性）。

    10 只等权组合上，「吃掉调仓日当天收益」的相对增益约 +20%，
    但配对 t 只有 1.4 —— 与噪声不可区分。必须报 OK 并给出 MDE。
    """
    rep = _run(wm_clean, px, la.check_rebalance_alignment)
    assert not rep.blockers and not rep.warnings, \
        [f.name for f in rep.findings if f.level != OK]
    f = [f for f in rep.findings if "调仓日" in f.name][0]
    assert "不显著" in f.detail or "无额外收益" in f.detail
    assert np.isfinite(rep.stats["day0_mde"])


def test_day0_reports_mde_so_null_result_is_interpretable(px, wm_clean):
    """负结果必须附最小可检出效应，否则「没查出问题」无法解读。"""
    rep = _run(wm_clean, px, la.check_rebalance_alignment)
    mde = rep.stats["day0_mde"]
    assert 0 < mde < 0.05
    assert abs(rep.stats["day0_t"]) < 2.0


def test_day0_skips_when_too_few_periods(px):
    reb = month_ends(px)[:4]
    wm = equal_weight(px, reb, k=8)
    rep = _run(wm, px, la.check_rebalance_alignment)
    assert rep.skipped and "调仓日对齐" in rep.skipped[0]


# ---------------- ② 权重前视 ----------------

def test_weight_lookahead_caught_when_planted(px, wm_dirty):
    """按下期收益加权 ⇒ 必须 BLOCK。"""
    rep = _run(wm_dirty, px, la.check_weight_lookahead)
    assert rep.blockers
    f = rep.blockers[0]
    assert "同期收益" in f.name
    assert rep.stats["weight_ret_corr"] > 0.9


def test_weight_lookahead_equal_weight_reports_undefined_not_short(px, wm_clean):
    """★ 等权 ⇒ 相关系数【无定义】，不能报成「持仓过少」。

    报错原因会让客户去修一个不存在的问题（factor-audit 实测教训：
    把复牌报成复权错误，客户从此不信后续所有告警）。
    """
    rep = _run(wm_clean, px, la.check_weight_lookahead)
    assert not rep.blockers and not rep.warnings
    f = [f for f in rep.findings if "权重前视" in f.name][0]
    assert "无定义" in f.detail and "等权" in f.detail
    # 绝不能出现误导性的跳过理由
    assert not any("持仓过少" in s for s in rep.skipped)


def test_weight_lookahead_clean_tilt_does_not_warn(px, reb):
    """权重有截面变化但只用【过去】信息 ⇒ 不该报警。"""
    wm = tilted_weight(px, reb, k=10, seed=9)
    rep = _run(wm, px, la.check_weight_lookahead)
    assert not rep.blockers, [f.name for f in rep.blockers]


def test_weight_lookahead_perfect_agreement_is_not_read_as_no_evidence():
    """★ 各期相关系数完全一致 ⇒ 期间方差为 0 ⇒ t 无定义。

    那是【最强】的证据，不是缺证据。第一版把 `not isfinite(t)` 判成 OK，
    于是 corr=+1.000、100% 期为正的植入前视被报成「无系统性关联」。
    这条测试把那个分支钉死。
    """
    px_d = make_prices(n_codes=30, n_dead=6, seed=3)
    reb = month_ends(px_d)
    wm = lookahead_weight(px_d, reb, k=8)
    rep = _run(wm, px_d, la.check_weight_lookahead)
    assert abs(rep.stats["weight_ret_corr"] - 1.0) < 1e-9, "构造未产生完美相关"
    assert rep.blockers, "完美一致的前视必须 BLOCK，不能落进通过分支"
    assert "无系统性关联" not in rep.blockers[0].detail


def test_weight_lookahead_t_display_does_not_overflow(px, wm_dirty):
    """完美相关时朴素算法会给出 t=5.2e16 这种没有信息量的数。

    显示必须换成人能读懂的说法，且不能出现科学计数法的溢出值。
    """
    rep = _run(wm_dirty, px, la.check_weight_lookahead)
    d = rep.blockers[0].detail
    assert "e+" not in d.lower(), d
    assert "无期间方差" in d or "t=" in d, d


# ---------------- ③ 股票池生存者偏差 ----------------

def test_survivorship_caught_when_universe_is_survivors_only():
    """价格面板含已消失标的，但持仓从不碰它们 ⇒ BLOCK。"""
    px_d = make_prices(n_codes=30, n_dead=8, seed=3)
    last = px_d.index.max()
    alive = [c for c in px_d.columns if np.isfinite(px_d.loc[last, c])]
    reb = month_ends(px_d)
    wm = equal_weight(px_d[alive], reb, k=8, seed=6)
    wm = wm.reindex(columns=px_d.columns, fill_value=0.0)
    rep = _run(wm, px_d, la.check_universe_survivorship)
    assert rep.blockers and "生存者" in rep.blockers[0].name


def test_survivorship_ok_when_dead_names_held():
    """持仓包含已退出标的 ⇒ OK。"""
    px_d = make_prices(n_codes=30, n_dead=8, seed=3)
    reb = month_ends(px_d)
    wm = equal_weight(px_d, reb, k=8, seed=2)
    rep = _run(wm, px_d, la.check_universe_survivorship)
    assert not rep.blockers, [f.name for f in rep.blockers]


def test_survivorship_cannot_judge_on_pure_survivor_panel(px, wm_clean):
    """价格面板本身没有退市股 ⇒ 必须说「无法判定」，不能报 OK。"""
    rep = _run(wm_clean, px, la.check_universe_survivorship)
    f = [f for f in rep.findings if "生存者" in f.name][0]
    assert f.level == WARN and "无法判定" in f.name


# ---------------- ④ 成分变动记账 ----------------

def test_membership_policy_spread_reported():
    """有缺价事件 ⇒ 必须报三种政策的净值区间。

    ★ 只断言「区间被报出来」和「清算最低」，【不】断言 drop 与 hold_last
    的高低 —— 那个次序只在退市前有崩盘时成立（见 test_core.py 两条）。
    报告本身也只给区间，不声称哪种政策偏高。
    """
    px_d = make_prices(n_codes=20, n_dead=6, seed=3, death_drawdown=0.8)
    reb = month_ends(px_d)
    wm = equal_weight(px_d, reb, k=8, seed=2)
    rep = AuditReport()
    out = la.check_membership_accounting(wm, px_d, rep)
    assert out["zero"] < out["hold_last"], out
    assert max(out.values()) > min(out.values()), "未产生政策差异，测试无效"
    f = [f for f in rep.findings if "记账" in f.name][0]
    assert "无损移除" in f.detail and "全额清算" in f.detail


def test_membership_ok_when_no_missing_prices(px, wm_clean):
    rep = AuditReport()
    la.check_membership_accounting(wm_clean, px, rep)
    f = [f for f in rep.findings if "记账" in f.name][0]
    assert f.level == OK and "没有任何一期" in f.detail


def test_membership_impact_states_both_wrong_directions():
    """报告必须同时给出两个错法的方向，否则客户会挑对自己有利的。"""
    px_d = make_prices(n_codes=20, n_dead=8, seed=3)
    reb = month_ends(px_d)
    wm = equal_weight(px_d, reb, k=6, seed=4)
    rep = AuditReport()
    la.check_membership_accounting(wm, px_d, rep)
    f = [f for f in rep.findings if "记账" in f.name][0]
    if f.level != OK:
        # 两个错法的机制都要说清楚，但【不】声称哪种政策必然偏高 ——
        # 那个次序只在退市前有崩盘时成立（见 test_core.py 的两条断言）
        assert "扔掉" in f.impact and "暴跌" in f.impact
        assert "权威退市日" in f.impact
        assert "虚高" not in f.impact, "不该给出无条件的方向断言"
