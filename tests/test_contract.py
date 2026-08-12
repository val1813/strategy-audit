"""输入契约：会让下游【静默算错】的输入必须 BLOCK。"""

from __future__ import annotations

import numpy as np
import pandas as pd

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
