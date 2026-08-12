"""敌意输入：不许崩，也不许交空报告。

★ 为什么「空报告」和「崩」一样严重
------------------------------
实测（10 种敌意输入）暴露两类问题：
  · 净值里一个 ±inf ⇒ NW t 的方差计算崩在
    `invalid value encountered in scalar divide`，用户不知道是哪行数据
  · 单行表 / 全 NaN / 净值全 0 ⇒ 返回 0 条 finding、0 个 BLOCK
    用户看到「什么都没说」，比报错更无从下手

所以这里的断言是：任何输入都必须得到【结论】或【明确的原因】，
二者之一，不能两者都没有。
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from strategy_audit import audit

IDX = pd.date_range("2020-01-31", periods=60, freq="ME")


def _cases():
    return {
        "单行表": pd.DataFrame({"date": ["2021-01-04"], "nav": [1.0]}),
        "全NaN": pd.DataFrame({"date": IDX, "nav": [np.nan] * 60}),
        "日期全重复": pd.DataFrame({"date": [IDX[0]] * 60,
                                "nav": np.linspace(1, 2, 60)}),
        "倒序日期": pd.DataFrame({"date": IDX[::-1],
                              "nav": np.linspace(1, 2, 60)}),
        "含inf": pd.DataFrame({"date": IDX, "nav": [1.0] * 59 + [np.inf]}),
        "含-inf": pd.DataFrame({"date": IDX, "nav": [1.0] * 59 + [-np.inf]}),
        "净值全0": pd.DataFrame({"date": IDX, "nav": [0.0] * 60}),
        "常数净值": pd.DataFrame({"date": IDX, "nav": [1.0] * 60}),
        "只有日期列": pd.DataFrame({"date": IDX}),
        "负净值": pd.DataFrame({"date": IDX, "nav": np.linspace(-1, 1, 60)}),
        "超大值": pd.DataFrame({"date": IDX, "nav": np.full(60, 1e300)}),
        "空表": pd.DataFrame(),
    }


@pytest.mark.parametrize("name", list(_cases()))
def test_hostile_input_never_crashes(name):
    """★ 连 numpy 的 RuntimeWarning 都不许有（那说明在算 inf/0）。"""
    df = _cases()[name]
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        rep = audit(df, show_detection=False)
    assert rep is not None


@pytest.mark.parametrize("name", list(_cases()))
def test_hostile_input_never_returns_silent_empty(name):
    """★ 要么给结论，要么给明确原因 —— 不许两者都没有。"""
    rep = audit(_cases()[name], show_detection=False)
    real = [f for f in rep.findings
            if f.section not in ("输入识别", "输入契约")]
    assert real or rep.blockers, (
        f"{name}：0 条实质结论且 0 个 BLOCK —— 用户看到一份空报告")


def test_infinite_values_are_reported_not_swallowed():
    """±inf 被丢弃这件事必须说出来，不能悄悄清掉。"""
    df = pd.DataFrame({"date": IDX, "nav": [1.0] * 58 + [np.inf, 1.1]})
    rep = audit(df, show_detection=False)
    assert any("无效值" in f.name for f in rep.findings)
    f = [f for f in rep.findings if "无效值" in f.name][0]
    assert "inf" in f.impact or "±inf" in f.impact


def test_reversed_dates_sorted_not_rejected():
    """倒序日期应当排好继续审，而不是报错。"""
    r = pd.Series(np.random.default_rng(3).normal(0.006, 0.04, 60), index=IDX)
    nav = (1 + r).cumprod()
    fwd = audit(pd.DataFrame({"date": IDX, "nav": nav.values}),
                show_detection=False)
    rev = audit(pd.DataFrame({"date": IDX[::-1], "nav": nav.values[::-1]}),
                show_detection=False)
    key = lambda x: sorted((f.level, f.name) for f in x.findings
                           if f.section == "策略层显著性")
    assert key(fwd) == key(rev)


def test_duplicate_dates_kept_last():
    """重复日期保留最后一条，且不该让检查数量变化。"""
    r = pd.Series(np.random.default_rng(4).normal(0.006, 0.04, 60), index=IDX)
    nav = (1 + r).cumprod()
    base = pd.DataFrame({"date": IDX, "nav": nav.values})
    dup = pd.concat([base, base.tail(3)], ignore_index=True)
    a = audit(base, show_detection=False)
    b = audit(dup, show_detection=False)
    key = lambda x: sorted(f.name for f in x.findings
                           if f.section == "策略层显著性")
    assert key(a) == key(b)


def test_empty_report_explains_minimum_input():
    """跑不起来时必须告诉用户最低需要什么。"""
    rep = audit(pd.DataFrame({"date": IDX}), show_detection=False)
    assert rep.blockers
    imp = " ".join(f.impact for f in rep.blockers)
    assert "净值曲线" in imp or "--demo" in imp


def test_no_input_at_all():
    rep = audit(show_detection=False)
    assert rep.blockers


def test_text_renders_for_every_hostile_case():
    """★ 报告必须能打印出来 —— 崩在 text() 上和崩在检查里一样糟。"""
    for name, df in _cases().items():
        rep = audit(df, show_detection=False)
        txt = rep.text()
        assert isinstance(txt, str) and len(txt) > 50, name
        assert "nan" not in txt.lower().replace("nan/", ""), (
            f"{name}：报告里漏出了 nan")
