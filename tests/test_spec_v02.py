"""Acceptance guards from SPEC v0.2."""
from __future__ import annotations

import numpy as np
import pandas as pd

from strategy_audit import audit
from strategy_audit.core import period_returns, price_matrix
from strategy_audit.contract import normalize_gross, to_matrix
from strategy_audit.parameter_audit import check_deferred_exit
from strategy_audit.report import AuditReport, OK, SKIP, WARN
from strategy_audit import parameter_audit as pa
from strategy_audit.lookahead import check_universe_survivorship

from synth import equal_weight, make_prices, month_ends, to_long


def _base():
    px = make_prices(n_codes=20)
    wm = equal_weight(px, month_ends(px), k=5)
    return px, wm, to_long(wm, "weight"), to_long(px, "close")


def test_identity_reported_as_skip():
    px, wm, w, p = _base()
    r = period_returns(wm, px)["ret"]
    nav = (1 + r).cumprod().rename("nav").rename_axis("date").reset_index()
    rep = audit(w, p, nav, show_detection=False)
    f = next(f for f in rep.findings if f.key == "nav_recon")
    assert f.level == SKIP
    assert "不可审" in f.detail and "同源" in f.detail


def test_open_execution_uses_open_and_never_falls_back():
    px, wm, w, p = _base()
    p["open"] = p["close"] * 0.99
    a = audit(w, p, execution={"entry": "open"}, show_detection=False)
    b = audit(w, p.drop(columns="open"), execution={"entry": "open"},
              show_detection=False)
    assert "nav" in a.stats["capability"]
    assert "nav" not in b.stats["capability"]
    assert any("不回退到 close" in s for s in b.skipped)


def test_sparse_panel_skips_survivorship():
    dates = pd.bdate_range("2024-01-01", periods=10)
    cols = [f"C{i}" for i in range(10)]
    pm = pd.DataFrame(np.nan, index=dates, columns=cols)
    pm.iloc[0, :] = 100.0
    pm.iloc[-1, 0] = 101.0
    wm = pd.DataFrame(0.1, index=dates[[0, 5, 9]], columns=cols)
    rep = AuditReport()
    check_universe_survivorship(wm, pm, rep)
    assert any("末日价格覆盖率" in s for s in rep.skipped)
    assert not any("已退市" in f.detail for f in rep.findings)


def test_family7_never_recommends():
    px, wm, w, p = _base()
    rep = audit(w, p, params={"holding_days": 3}, show_detection=False)
    family = [f for f in rep.findings if f.section == "参数邻域体检"]
    assert family
    for f in family:
        for banned in ("建议改为", "最优", "推荐使用", "应当改成"):
            assert banned not in f.detail + f.impact


def test_holding_neighborhood_is_trade_level_not_rebalance_level():
    dates = pd.bdate_range("2024-01-01", periods=12)
    codes = ["A", "B"]
    close = pd.DataFrame({"A": np.arange(100, 112),
                          "B": np.full(12, 100.0)}, index=dates)
    open_ = close.copy()
    # One event per day, with a large cash-heavy rebalance snapshot. The
    # event-level result must still have one observation per entry, not one
    # observation per rebalance date weighted by the cash residue.
    wm = pd.DataFrame(0.0, index=dates[:8], columns=codes)
    for i, d in enumerate(wm.index):
        wm.loc[d, codes[i % 2]] = 0.01
    trades = pd.DataFrame({"entry_date": dates[:8], "code": codes * 4})
    one = pa._event_returns(wm, open_, close, 1, trades)
    three = pa._event_returns(wm, open_, close, 3, trades)
    assert len(one) == 8 and len(three) == 8
    z = pa._paired(three, one)
    assert z["n"] == 8


def test_h_counts_entry_day_and_matches_actual_exit():
    dates = pd.bdate_range("2024-01-01", periods=10)
    pm = pd.DataFrame({"A": np.linspace(100, 109, len(dates))}, index=dates)
    wm = pd.DataFrame(1.0, index=dates[:5], columns=["A"])
    trades = pd.DataFrame({"entry_date": dates[:5], "code": ["A"] * 5,
                           "exit_date": dates[2:7]})
    actual = pa._event_returns(wm, pm, pm, 3, trades, use_actual_exit=True)
    declared = pa._event_returns(wm, pm, pm, 3, trades)
    assert len(actual) == len(declared) == 5
    assert np.allclose(actual.values, declared.values)


def test_h_one_is_entry_session_not_next_session():
    dates = pd.bdate_range("2024-01-01", periods=3)
    pm = pd.DataFrame({"A": [100.0, 110.0, 120.0]}, index=dates)
    wm = pd.DataFrame(1.0, index=[dates[0]], columns=["A"])
    r = pa._event_returns(wm, pm, pm, 1)
    assert r.iloc[0] == 0.0


def test_holding_split_half_is_invariant_to_trade_row_order():
    dates = pd.bdate_range("2024-01-01", periods=120)
    # First 60 events improve from H=1 to H=3; the last 60 also improve.
    # Deliberately scramble the supplied trade table, as real exports often
    # arrive sorted by realized return rather than entry date.
    pm = pd.DataFrame({"A": 100.0 + np.arange(len(dates)),
                       "B": 200.0 + np.arange(len(dates))}, index=dates)
    wm = pd.DataFrame(0.0, index=dates[:100], columns=["A", "B"])
    ordered = pd.DataFrame({"entry_date": dates[:100],
                            "code": ["A" if i % 2 == 0 else "B" for i in range(100)]})
    shuffled = ordered.sample(frac=1.0, random_state=73).reset_index(drop=True)
    a = pa._event_returns(wm, pm, pm, 3, ordered)
    b = pa._event_returns(wm, pm, pm, 1, ordered)
    x = pa._event_returns(wm, pm, pm, 3, shuffled)
    y = pa._event_returns(wm, pm, pm, 1, shuffled)
    z_ordered, z_shuffled = pa._paired(a, b), pa._paired(x, y)
    assert z_ordered["halves_same"] and z_shuffled["halves_same"]
    assert z_ordered["passed"] == z_shuffled["passed"]
    assert np.isclose(z_ordered["t"], z_shuffled["t"])


def test_topn_is_explicitly_display_only():
    px, wm, w, p = _base()
    sig = pd.DataFrame({"date": px.index[::5].repeat(len(px.columns)),
                        "code": list(px.columns) * len(px.index[::5]),
                        "score": np.arange(len(px.index[::5]) * len(px.columns))})
    rep = audit(w, p, signals=sig, params={"holding_days": 3, "top_n": 1},
                show_detection=False)
    f = next(f for f in rep.findings if "Top-N" in f.name)
    assert f.level == SKIP
    assert "零模型" in f.impact and "MDE" in f.impact


def test_open_execution_changes_recomputed_return():
    dates = pd.bdate_range("2024-01-01", periods=4)
    w = pd.DataFrame({"date": [dates[0], dates[2]], "code": ["A", "A"],
                      "weight": [1.0, 1.0]})
    p = pd.DataFrame({"date": dates, "code": ["A"] * 4,
                      "close": [100.0, 110.0, 120.0, 130.0],
                      "open": [90.0, 95.0, 100.0, 105.0]})
    wm = normalize_gross(to_matrix(w))
    pm_close, pm_open = price_matrix(p), price_matrix(p, "open")
    close_ret = period_returns(wm, pm_close)["ret"]
    open_ret = period_returns(wm, pm_close, entry_pm=pm_open)["ret"]
    assert not np.allclose(close_ret.values, open_ret.values)


def test_pure_export_reports_family7_as_unavailable():
    _, _, w, p = _base()
    rep = audit(w, p, show_detection=False)
    family_skips = [s for s in rep.skipped if any(name in s for name in
                    ("持有期邻域", "Top-N", "入场时点", "信号-持仓", "跨供应商",
                     "跌停出场顺延"))]
    assert len(family_skips) == 6
    assert "缺" in "".join(family_skips)


# ---- 跌停出场顺延：判板必须用行情价，不用成交价 ----

def _limit_down_case():
    """一笔出场撞在跌停封板日：exit close 精确等于 down_limit。"""
    dates = pd.bdate_range("2024-01-01", periods=6)
    close = [100.0, 105.0, 94.5, 90.0, 95.0, 96.0]   # idx2 = 105*0.9 封板
    dl = [90.0, 94.5, 94.5, 85.0, 85.5, 86.4]        # idx2 down_limit == close
    p = pd.DataFrame({"date": dates, "code": ["A"] * 6, "close": close,
                      "down_limit": dl})
    trades = pd.DataFrame({"entry_date": [dates[0]], "code": ["A"],
                          "exit_date": [dates[2]]})
    return p, trades, dates


def test_deferred_exit_uses_market_price_not_fill_price():
    """★ 核心回归：行情价判板命中，成交价判板必然漏报。

    真实客户件上第一版用 `实际卖出价 == down_limit` 得 0/796，漏掉 8 笔
    真封板（占总盈亏 −13.41%）。成交价含滑点，等式在构造上不成立。
    """
    p, trades, dates = _limit_down_case()
    rep = AuditReport()
    close_pm = p.pivot(index="date", columns="code", values="close")
    dl_pm = p.pivot(index="date", columns="code", values="down_limit")
    check_deferred_exit(trades, close_pm, {"down_limit": dl_pm}, rep)
    f = next(f for f in rep.findings if "跌停" in f.name)
    assert f.level == WARN
    assert rep.stats["deferred_exit_n"] == 1
    assert "market" not in f.detail  # 报文是中文
    assert "成交价" in f.detail      # 必须写明判据用行情价不用成交价

    # 成交价（含滑点）判板：同一笔必然不命中 —— 这是漏报的成因
    fill_price = 88.7  # 真实成交价，永不精确等于 90.0 的跌停价
    assert not np.isclose(fill_price, dl_pm.iloc[2, 0], rtol=1e-6, atol=1e-6)


def test_deferred_exit_entry_basis_matches_execution():
    """入场价口径必须跟随 execution；否则同一报告出现两个矛盾均值。"""
    p, trades, dates = _limit_down_case()
    p["open"] = p["close"] * 0.5          # 夸张的 open，便于区分
    close_pm = p.pivot(index="date", columns="code", values="close")
    open_pm = p.pivot(index="date", columns="code", values="open")
    dl_pm = p.pivot(index="date", columns="code", values="down_limit")
    r_close, r_open = AuditReport(), AuditReport()
    check_deferred_exit(trades, close_pm, {"down_limit": dl_pm}, r_close)
    check_deferred_exit(trades, close_pm, {"down_limit": dl_pm}, r_open,
                        entry_pm=open_pm)
    assert not np.isclose(r_close.stats["deferred_exit_asis_mean"],
                          r_open.stats["deferred_exit_asis_mean"])


def test_deferred_exit_clean_case_is_ok_not_silent():
    """没有封板时必须显式报 OK 并写明判据，不能静默跳过。"""
    dates = pd.bdate_range("2024-01-01", periods=4)
    p = pd.DataFrame({"date": dates, "code": ["A"] * 4,
                      "close": [100.0, 101.0, 102.0, 103.0],
                      "down_limit": [90.0, 90.9, 91.8, 92.7]})
    trades = pd.DataFrame({"entry_date": [dates[0]], "code": ["A"],
                          "exit_date": [dates[2]]})
    rep = AuditReport()
    check_deferred_exit(trades, p.pivot(index="date", columns="code", values="close"),
                        {"down_limit": p.pivot(index="date", columns="code",
                                               values="down_limit")}, rep)
    f = next(f for f in rep.findings if "跌停" in f.name)
    assert f.level == OK and "成交价" in f.detail


def test_deferred_exit_skips_without_down_limit():
    p, trades, _ = _limit_down_case()
    rep = AuditReport()
    check_deferred_exit(trades, p.pivot(index="date", columns="code", values="close"),
                        {}, rep)
    assert any("down_limit" in s for s in rep.skipped)
    assert not rep.findings


def test_trades_without_exit_date_is_not_counted_as_available():
    """只有 entry_date/code 的交易明细不足以审顺延 —— 不许显示成能审。"""
    _, _, w, p = _base()
    p["down_limit"] = p["close"] * 0.9
    tr = pd.DataFrame({"entry_date": [p["date"].iloc[0]], "code": [p["code"].iloc[0]]})
    rep = audit(w, p, trades=tr, show_detection=False)
    assert any("exit_date" in s or "跌停出场顺延" in s for s in rep.skipped)
