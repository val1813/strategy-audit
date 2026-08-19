"""输入契约：会让下游【静默算错】的输入必须 BLOCK。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from strategy_audit import contract
from strategy_audit.report import BLOCK, WARN, AuditReport

from synth import equal_weight, make_prices, month_ends, to_long


def _w_long(px, reb):
    return to_long(equal_weight(px, reb), "weight")


def test_missing_columns_block():
    rep = AuditReport()
    out = contract.load_weights(pd.DataFrame({"date": [], "code": []}), rep)
    assert out is None
    assert rep.blockers and "必需列" in rep.blockers[0].name


def test_null_weight_blocks_not_filled(px, reb):
    """★ null 权重必须 BLOCK，不能 fillna(0)。

    「不持有」(=0) 与「没取到」(未知) 是两种含义。当成 0 会把数据缺失
    记成主动清仓 —— 凭空造出一笔换手。
    """
    w = _w_long(px, reb)
    w.loc[w.index[:5], "weight"] = np.nan
    rep = AuditReport()
    assert contract.load_weights(w, rep) is None
    assert any("空值" in f.name for f in rep.blockers)
    # 报告必须解释清楚两种含义，否则客户会自己 fillna(0)
    assert "清仓" in rep.blockers[0].impact


def test_duplicate_weight_rows_block(px, reb):
    w = _w_long(px, reb)
    rep = AuditReport()
    assert contract.load_weights(pd.concat([w, w.head(3)]), rep) is None
    assert any("重复" in f.name for f in rep.blockers)


def test_duplicate_price_rows_block(px, reb):
    """实测踩过的缺陷：真实面板某月两日共 11998 重复行。"""
    p = to_long(px, "close")
    rep = AuditReport()
    assert contract.load_prices(pd.concat([p, p.head(4)]), rep) is None
    assert any("重复" in f.name for f in rep.blockers)


def test_short_weights_warn_not_block(px, reb):
    """空头不是错误，但换手口径不同 ⇒ WARN。"""
    w = _w_long(px, reb)
    w.loc[w.index[0], "weight"] = -0.1
    rep = AuditReport()
    assert contract.load_weights(w, rep) is not None
    assert any("空头" in f.name and f.level == WARN for f in rep.findings)


def test_gross_one_is_ok(px, reb):
    w = contract.load_weights(_w_long(px, reb), AuditReport())
    rep = AuditReport()
    contract.check_gross(w, rep)
    assert not rep.blockers and not rep.warnings


def test_gross_off_by_5pct_reports_both_readings(px, reb):
    """权重和 0.95 有两种含义，工具必须都说出来而不是替客户判断。"""
    w = _w_long(px, reb)
    w["weight"] = w["weight"] * 0.95
    rep = AuditReport()
    contract.check_gross(contract.load_weights(w, AuditReport()), rep)
    f = [f for f in rep.findings if "权重和" in f.name][0]
    assert f.level == BLOCK          # 0.05 > GROSS_BLOCK
    assert "现金" in f.impact and "算错" in f.impact


def test_gross_tiny_deviation_warns_only(px, reb):
    w = _w_long(px, reb)
    w["weight"] = w["weight"] * (1 + 1e-4)
    rep = AuditReport()
    contract.check_gross(contract.load_weights(w, AuditReport()), rep)
    assert not rep.blockers
    assert any("权重和" in f.name for f in rep.warnings)


def test_normalize_gross_makes_rows_sum_to_one(px, reb):
    m = contract.to_matrix(contract.load_weights(_w_long(px, reb),
                                                 AuditReport()))
    n = contract.normalize_gross(m * 0.4)
    assert np.allclose(n.abs().sum(axis=1).values, 1.0, atol=1e-12)


def test_to_matrix_absent_row_means_zero(px, reb):
    """长表里没有的 (date, code) = 不持有，填 0 是对的（null 已在上游拦掉）。"""
    m = contract.to_matrix(contract.load_weights(_w_long(px, reb),
                                                 AuditReport()))
    assert (m.values == 0.0).any()
    assert not m.isna().any().any()


# ---------------- 自报净值对账 ----------------
#
# ★ 这一项是盲测抓出来的硬伤。第一版 _api.py 里是 elif：权重+价格在手
# 就自己重算收益曲线，客户交上来的那条净值【连看一眼都没有】。
# 盲测一份四因子月频策略实测两条差 6.0%（自报 6.19% / 重算 6.66%，
# 原因是月内日频再平衡，权重表里看不出来）—— 所有检查都算在一条
# 客户从未汇报过的净值上，而报告照样打印得像模像样。

def _rets(vals, start="2016-01-31"):
    idx = pd.date_range(start, periods=len(vals), freq="ME")
    return pd.Series(vals, index=idx)


@pytest.mark.parametrize("anchor", ["before", "at_first"])
def test_nav_recon_identical_curves_pass(anchor):
    """★ 两种净值写法都必须报 OK —— 误报比漏报更糟。

    净值序列有两种常见写法，从数值上无法区分：
        anchor="before"    起点锚在 1.0，位于第一个调仓日【之前】
        anchor="at_first"  起点就是第一期末（首期收益已算进起点）
    第一版盲取首末，于是后一种写法上凭空多算一期收益，
    一个干净策略被报 1.0% 偏差。报警一旦不可信就没人看了。
    """
    r = _rets(np.full(36, 0.01))
    nav = (1.0 + r).cumprod()
    if anchor == "before":
        nav = pd.concat([pd.Series([1.0],
                                   index=[r.index[0] - pd.Timedelta(days=1)]),
                         nav])
    rep = AuditReport()
    out = contract.check_nav_reconciliation(r, nav, "nav", rep)
    assert abs(out["rel"]) < 1e-9, (anchor, out)
    assert not rep.blockers and not rep.warnings
    assert "一致" in rep.findings[0].detail


def test_nav_recon_daily_curve_starting_late():
    """★ 日频自报净值起点晚于首个调仓日（月末遇周末时必然如此）。

    第一版在这里直接判「覆盖不到调仓区间」而跳过 —— 一个最常见的
    真实输入被当成无法对账。现在取两条曲线【共同覆盖】的调仓日窗口。
    """
    r = _rets(np.full(36, 0.01))
    didx = pd.bdate_range(r.index[0] + pd.Timedelta(days=1), periods=36 * 21)
    daily = pd.Series(np.cumprod(1 + np.full(len(didx), 0.01 / 21)), index=didx)
    rep = AuditReport()
    out = contract.check_nav_reconciliation(r, daily, "nav", rep)
    assert out, "常见的日频净值输入不该被跳过"
    assert abs(out["rel"]) < 0.05
    assert "nav_recon_window" in rep.stats


def test_nav_recon_partial_overlap_uses_common_window():
    """自报曲线只覆盖前半段 ⇒ 在公共窗口上对账，而不是拿全样本比半样本。"""
    r = _rets(np.full(36, 0.01))
    rep = AuditReport()
    out = contract.check_nav_reconciliation(r, (1.0 + r).cumprod().iloc[:18],
                                            "nav", rep)
    assert abs(out["rel"]) < 1e-9, out
    assert not rep.blockers


def test_nav_recon_no_overlap_skips():
    """完全不重叠必须跳过并写明两段区间，不能报 OK 也不能崩。"""
    r = _rets(np.full(36, 0.01))
    off = pd.Series([1.0, 1.1, 1.2],
                    index=pd.date_range("2030-01-31", periods=3, freq="ME"))
    rep = AuditReport()
    assert contract.check_nav_reconciliation(r, off, "nav", rep) == {}
    assert rep.skipped and not rep.findings
    assert "2030" in rep.skipped[0]


def test_nav_recon_self_report_too_high_blocks():
    """★ 危险方向：自报【高于】重算 ⇒ 权重表算不出你汇报的收益 ⇒ BLOCK。

    这是盲测那份策略的真实情形（自报 +122.38% / 重算 +112.15%，差 4.6%）。
    这个方向不可能由「扣成本」解释 —— 扣成本只会让自报更低。
    """
    r = _rets(np.full(36, 0.004))
    own = (1.0 + r * 3.0).cumprod()          # 自报远高于权重表能解释的
    rep = AuditReport()
    out = contract.check_nav_reconciliation(r, own, "nav", rep, turnover=0.5)
    assert out["rel"] < 0
    assert rep.blockers, [f.name for f in rep.findings]
    txt = rep.findings[0].impact
    assert "权重表【算不出】" in txt
    assert "不可能由「扣成本」解释" in txt


@pytest.mark.parametrize("bp,expect_ok", [
    (10, True), (25, True), (50, True), (150, False), (400, False),
])
def test_nav_recon_cost_deduction_is_not_a_defect(bp, expect_ok):
    """★ 门槛必须【不对称】：老老实实扣了成本不能被判成缺陷。

    重算是毛口径，所以扣过成本的自报净值【必然】低于重算。用对称门槛
    会把最普通的正确实践报成 BLOCK：单边 25bp、年换手 8x、十年样本，
    累计差轻松超过 5%。误报做对的人比漏报更糟。

    做法是把缺口折算成隐含单边成本，只有大到不像成本时才升级。
    """
    r = _rets(np.full(120, 0.008))
    tau = 0.68
    own = (1.0 + (r - 2 * bp * 1e-4 * tau)).cumprod()
    rep = AuditReport()
    out = contract.check_nav_reconciliation(r, own, "nav", rep, turnover=tau)
    assert out["rel"] > 0                      # 良性方向
    if expect_ok:
        # 现实费率下，隐含成本要能大致还原出真实费率（同一量级）。
        # ★ 只在合理区间内断言还原精度：成本一旦超过毛收益，
        # 自报净值就转为衰减，rel 随期数指数放大，而折算用的是
        # 「每期缺口 ≈ rel/n」的一阶近似 —— 那里本来就不该准。
        # 判据是「像不像成本」，不是「精确等于多少」。
        assert 0.3 * bp <= out["implied_cost_bp"] <= 3.0 * bp, out
    else:
        assert out["implied_cost_bp"] > contract.NAV_RECON_COST_IMPLAUSIBLE_BP
    assert not rep.blockers
    if expect_ok:
        assert not rep.warnings, (bp, rep.findings[0].name)
        assert "不是缺陷" in rep.findings[0].impact
    else:
        assert rep.warnings, bp


def test_nav_recon_no_turnover_does_not_claim_magnitude():
    """★ 算不出隐含成本时不许暗示「量级偏大」—— 那是我们没能算。"""
    r = _rets(np.full(120, 0.008))
    own = (1.0 + (r - 2 * 25e-4 * 0.68)).cumprod()
    rep = AuditReport()
    contract.check_nav_reconciliation(r, own, "nav", rep, turnover=None)
    f = rep.findings[0]
    assert "无法折算" in f.detail
    assert "无法判断量级" in f.impact
    assert "量级偏大" not in f.impact


def test_nav_recon_accepts_return_series_too():
    """自报的是【收益率】序列时口径要自动对上（不能当成净值取首末比）。"""
    r = _rets(np.full(24, 0.01))
    rep = AuditReport()
    contract.check_nav_reconciliation(r, r.copy(), "ret", rep)
    assert not rep.blockers and not rep.warnings


def test_nav_recon_short_series_skips_not_passes():
    """★ 序列太短要跳过并记录，不能报 OK（跳过 ≠ 查过通过）。"""
    rep = AuditReport()
    contract.check_nav_reconciliation(_rets([0.01, 0.02]),
                                      _rets([0.01, 0.02]), "ret", rep)
    assert rep.skipped and not rep.findings


def test_nav_recon_is_registered_and_needs_own_nav():
    """★ 对账项必须声明需要 OWN_NAV 而不是 NAV。

    NAV 在权重+价格齐全时是工具【自己重算】的，所以它总是「有」。
    若对账项声明成 needs=(W,P,NAV)，能力矩阵就会在客户【没交净值】时
    把它显示成「能审」—— 把「没查」显示成「查过了」，这是本工具
    最忌讳的失败模式。
    """
    from strategy_audit import capability as cap
    c = next(c for c in cap.CHECKS if c.key == "nav_recon")
    assert cap.OWN_NAV in c.needs
    assert cap.NAV not in c.needs
    # 只有权重+价格（工具自己能重算 NAV）时，对账项必须显示为【审不了】
    ok, no = cap.available({cap.W, cap.P, cap.NAV})
    assert "nav_recon" in [x.key for x in no]
    ok2, _ = cap.available({cap.W, cap.P, cap.NAV, cap.OWN_NAV})
    assert "nav_recon" in [x.key for x in ok2]
