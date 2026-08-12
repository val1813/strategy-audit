"""输入识别：用户给什么我们都得认出来，认不出来要说认不出来。

★ 识别错比识别不出更糟
--------------------
把价格当净值会让所有检查算在错的东西上，而报告照样打印得像模像样。
所以这里既测「该认出来的认出来」，也测「不确定时要说不确定」。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from strategy_audit import detect
from strategy_audit.detect import classify_series, detect_frame, looks_like_percent

from synth import equal_weight, make_prices, month_ends, to_long


# ---------------- 列名别名 ----------------

@pytest.mark.parametrize("col,kind", [
    ("date", "date"), ("trade_date", "date"), ("日期", "date"),
    ("交易日期", "date"), ("dt", "date"),
    ("code", "code"), ("ts_code", "code"), ("股票代码", "code"),
    ("证券代码", "code"), ("symbol", "code"),
    ("weight", "weight"), ("目标权重", "weight"), ("持仓", "weight"),
    ("close", "close"), ("收盘价", "close"), ("adj_close", "close"),
    ("nav", "nav"), ("累计净值", "nav"), ("资金曲线", "nav"),
    ("ret", "ret"), ("收益率", "ret"), ("pct_chg", "ret"),
])
def test_alias_matches(col, kind):
    assert detect.match_column([col], kind) == col


def test_unknown_column_not_matched():
    assert detect.match_column(["foo", "bar"], "weight") is None


# ---------------- 长表 ----------------

def test_long_weights_detected(px, wm_clean):
    d = detect_frame(to_long(wm_clean, "weight"))
    assert d.kind == "weights"
    assert list(d.frame.columns) == ["date", "code", "weight"]


def test_long_prices_detected(px):
    d = detect_frame(to_long(px, "close"))
    assert d.kind == "prices"
    assert list(d.frame.columns) == ["date", "code", "close"]


def test_chinese_columns_detected(px, wm_clean):
    w = to_long(wm_clean, "weight").rename(
        columns={"date": "调仓日期", "code": "证券代码", "weight": "目标权重"})
    d = detect_frame(w)
    assert d.kind == "weights"
    assert d.columns["weight"] == "目标权重"


def test_integer_dates_parsed(px, wm_clean):
    """20210104 这种整数日期必须能认。"""
    w = to_long(wm_clean, "weight")
    w["date"] = w["date"].dt.strftime("%Y%m%d").astype(int)
    d = detect_frame(w)
    assert d.kind == "weights"
    assert pd.api.types.is_datetime64_any_dtype(d.frame["date"])


def test_unnamed_value_column_guessed_by_gross(px, wm_clean):
    """列名认不出时，用「每期求和≈1」判权重。"""
    w = to_long(wm_clean, "weight").rename(columns={"weight": "x7"})
    d = detect_frame(w)
    assert d.kind == "weights", d.notes


def test_unnamed_price_column_not_called_weight(px):
    w = to_long(px, "close").rename(columns={"close": "x7"})
    d = detect_frame(w)
    assert d.kind == "prices", d.notes


# ---------------- 宽表 ----------------

def test_wide_weight_matrix_detected(px, wm_clean):
    d = detect_frame(wm_clean)
    assert d.kind == "weights" and d.layout == "wide"
    # 宽表转长表后不该保留 0 权重行
    assert (d.frame["weight"] != 0).all()


def test_wide_price_matrix_detected(px):
    d = detect_frame(px)
    assert d.kind == "prices" and d.layout == "wide"


# ---------------- 单列序列 ----------------

def test_nav_vs_returns_by_values():
    idx = pd.date_range("2020-01-31", periods=60, freq="ME")
    r = pd.Series(np.random.default_rng(1).normal(0.005, 0.04, 60), index=idx)
    nav = (1 + r).cumprod()
    assert classify_series(r)[0] == "ret"
    assert classify_series(nav)[0] == "nav"


def test_negative_values_never_nav():
    s = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02])
    kind, why = classify_series(s)
    assert kind == "ret"
    assert any("负值" in w for w in why)


def test_percent_form_flagged():
    """★ 收益率写成百分数（1.5 表示 1.5%）是最常见的静默口径错。"""
    idx = pd.date_range("2020-01-31", periods=60, freq="ME")
    r = pd.Series(np.random.default_rng(2).normal(0.5, 4.0, 60), index=idx)
    assert looks_like_percent(r)
    d = detect_frame(pd.DataFrame({"date": idx, "收益率": r.values}))
    assert d.kind == "series"
    assert any("百分数" in n for n in d.notes)


def test_normal_returns_not_flagged_as_percent():
    idx = pd.date_range("2020-01-31", periods=60, freq="ME")
    r = pd.Series(np.random.default_rng(3).normal(0.005, 0.04, 60), index=idx)
    assert not looks_like_percent(r)


def test_name_vs_values_conflict_reported():
    """列名叫 nav 但数值像收益率 ⇒ 必须提示冲突，不能闷着按名字走。"""
    idx = pd.date_range("2020-01-31", periods=60, freq="ME")
    r = pd.Series(np.random.default_rng(4).normal(0.005, 0.04, 60), index=idx)
    d = detect_frame(pd.DataFrame({"date": idx, "nav": r.values}))
    assert any("但数值特征更像" in n for n in d.notes)


def test_date_index_without_date_column():
    """索引是日期、没有 date 列 ⇒ 也要能认。"""
    idx = pd.date_range("2020-01-31", periods=40, freq="ME")
    s = pd.DataFrame({"净值": (1 + pd.Series(
        np.random.default_rng(5).normal(0.005, 0.03, 40))).cumprod().values},
        index=idx)
    d = detect_frame(s)
    assert d.kind == "series" and d.role == "nav"


# ---------------- 认不出来 ----------------

def test_no_date_column_is_unknown():
    d = detect_frame(pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}))
    assert d.kind == "unknown"
    assert any("日期" in n for n in d.notes)


def test_empty_frame_is_unknown():
    assert detect_frame(pd.DataFrame()).kind == "unknown"


def test_unparseable_dates_dropped_and_noted():
    d = detect_frame(pd.DataFrame({
        "date": ["2021-01-04", "not-a-date", "2021-01-06"],
        "nav": [1.0, 1.01, 1.02]}))
    assert any("无法解析" in n for n in d.notes)
