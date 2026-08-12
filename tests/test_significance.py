"""族三：策略层显著性。只要一条曲线就能跑。

★ 这个文件的重点是【误报率】
--------------------------
族三的四项都是「统计上看着可疑」类的检查，最容易做成一堆噪声告警。
第一版实测：集中度固定门槛 0.50 误报 61%（零分布中位数本身就是 0.53）、
单期占比门槛 0.30 误报 27%、NW 跨线 BLOCK 误报 16%。
所以这里每一项都同时测「干净序列不报警」和「植入缺陷能检出」。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from strategy_audit import significance as sg
from strategy_audit.core import annualize
from strategy_audit.report import BLOCK, OK, WARN, AuditReport

IDX = pd.date_range("2016-01-31", periods=120, freq="ME")


def _noise(seed, mu=0.006, sd=0.045, n=120):
    return pd.Series(np.random.default_rng(seed).normal(mu, sd, n),
                     index=pd.date_range("2016-01-31", periods=n, freq="ME"))


def _level(fn, r, **kw):
    rep = AuditReport()
    fn(r, rep, **kw)
    return (rep.findings[0].level if rep.findings else "SKIP"), rep


# ---------------- 对账：两条独立的显著性估计必须一致 ----------------

@pytest.mark.parametrize("seed", [3, 11, 7, 21])
def test_sharpe_se_agrees_with_nw_t(seed):
    """★ Sharpe/SE 与 NW lag=0 的 t 测的是同一件事，必须对得上。

    第一版把周期口径的方差项和年化口径的 Sharpe 混在一个式子里，
    算出 t=3.43 而 NW t 只有 1.02 —— 夸大 3.4 倍。
    两个独立估计对不上，就说明有一个错了。
    """
    r = _noise(seed)
    sr = annualize(r, 12.0)["sharpe"]
    se = sg.deflated_sharpe(sr, len(r), 1, 12.0)["se"]
    t_sharpe = sr / se
    t_nw = sg.newey_west_t(r.values, 0)
    assert abs(t_sharpe - t_nw) / abs(t_nw) < 0.05, (
        f"Sharpe/SE t={t_sharpe:.2f} 与 NW t={t_nw:.2f} 不一致")


def test_newey_west_zero_lag_equals_plain_t():
    r = _noise(5)
    plain = float(r.mean() / (r.std(ddof=0) / np.sqrt(len(r))))
    assert abs(sg.newey_west_t(r.values, 0) - plain) < 1e-9


# ---------------- ① 年份集中度 ----------------

def test_concentration_low_false_positive_on_noise():
    """★ 干净噪声的报警率必须低。第一版固定门槛 0.50 误报 61%。"""
    bad = 0
    N = 40
    for s in range(N):
        lvl, _ = _level(sg.check_year_concentration, _noise(1000 + s))
        bad += lvl not in (OK, "SKIP")
    assert bad / N <= 0.20, f"误报率 {bad / N:.0%} 过高"


def test_concentration_detects_real_concentration():
    """两年吃掉全部收益 ⇒ 必须报出来。"""
    hit = 0
    N = 20
    for s in range(N):
        r = _noise(2000 + s, mu=0.0, sd=0.04)
        r.iloc[12:36] += 0.055
        lvl, _ = _level(sg.check_year_concentration, r)
        hit += lvl not in (OK, "SKIP")
    assert hit / N >= 0.8, f"检出率只有 {hit / N:.0%}"


def test_concentration_skipped_when_too_few_years():
    """★ 年数太少时必须跳过，不能报 OK。

    3 个年度时零分布中位数就是 100%（最好两年当然占满正收益），
    任何策略都「与随机打散不可区分」—— 报 OK 是个假的通过。
    """
    r = _noise(7, n=30)
    lvl, rep = _level(sg.check_year_concentration, r)
    assert lvl == "SKIP"
    assert any("年度" in s and "区分力" in s for s in rep.skipped)


def test_concentration_reports_null_distribution():
    """报告必须给出置换零分布，否则用户无法判断 40% 算不算高。"""
    r = _noise(11)
    lvl, rep = _level(sg.check_year_concentration, r)
    f = rep.findings[0]
    assert "置换零分布" in f.detail and "分位" in f.detail


def test_concentration_deterministic():
    """★ 同一份输入必须给同一份报告（置换用固定种子）。"""
    r = _noise(11)
    a = _level(sg.check_year_concentration, r)[1].stats["year_conc_pctile"]
    b = _level(sg.check_year_concentration, r)[1].stats["year_conc_pctile"]
    assert a == b


# ---------------- ② NW lag 敏感性 ----------------

def test_nw_ok_when_both_ends_significant():
    """★ t 从 3.3 到 6.9 比值 2.1，但两端都显著 ⇒ 结论没变，不该报警。"""
    r = _noise(4, mu=0.02, sd=0.04)
    lvl, rep = _level(sg.check_nw_lag_sensitivity, r)
    ts = rep.stats["nw_t_by_lag"]
    assert min(abs(v) for v in ts.values()) >= 2.0, "构造未做到两端都显著"
    assert lvl == OK


def test_nw_low_false_positive_on_noise():
    """第一版一跨过 |t|=2 就 BLOCK，误报 16%。"""
    bad = 0
    N = 60
    for s in range(N):
        lvl, _ = _level(sg.check_nw_lag_sensitivity, _noise(5000 + s))
        bad += lvl not in (OK, "SKIP")
    assert bad / N <= 0.10, f"误报率 {bad / N:.0%} 过高"


def test_nw_detects_conclusion_flip():
    """构造一个真的跨线且跨得远的序列 ⇒ 必须报警。"""
    e = np.random.default_rng(3).normal(0, 0.04, 120)
    x = np.zeros(120)
    for i in range(1, 120):
        x[i] = 0.75 * x[i - 1] + e[i]
    r = pd.Series(x + 0.02, index=IDX)
    lvl, rep = _level(sg.check_nw_lag_sensitivity, r)
    ts = rep.stats["nw_t_by_lag"]
    lo, hi = min(abs(v) for v in ts.values()), max(abs(v) for v in ts.values())
    if lo < 2.0 <= hi and (hi - lo) >= sg.NW_CROSS_MIN_GAP:
        assert lvl == WARN


def test_nw_reports_full_range_not_single_t():
    r = _noise(9)
    lvl, rep = _level(sg.check_nw_lag_sensitivity, r)
    assert "lag 0~" in rep.findings[0].detail
    assert len(rep.stats["nw_t_by_lag"]) >= 3


# ---------------- ③ 多重检验折扣 ----------------

def test_trials_default_one_is_flagged_as_assumption():
    """★ 默认 n_trials=1 必须说明「这是假设不是事实」。

    静默假设 1 等于替用户隐瞒多重检验。
    """
    rep = AuditReport()
    sg.check_deflated_sharpe(_noise(11), 12.0, rep, n_trials=1)
    f = rep.findings[0]
    assert f.level == WARN
    assert "默认假设，不是事实" in f.detail


def test_more_trials_lower_adjusted_sharpe():
    r = _noise(11)
    prev = None
    for n in (1, 5, 20, 100):
        rep = AuditReport()
        sg.check_deflated_sharpe(r, 12.0, rep, n_trials=n)
        adj = rep.stats["sharpe_adjusted"]
        if prev is not None:
            assert adj <= prev, f"n_trials={n} 时折扣后 Sharpe 没有下降"
        prev = adj


def test_enough_trials_kills_the_sharpe():
    """试足够多次 ⇒ 折扣后不为正 ⇒ BLOCK。"""
    rep = AuditReport()
    sg.check_deflated_sharpe(_noise(11), 12.0, rep, n_trials=5000)
    assert rep.blockers
    assert "折扣后 Sharpe 不为正" in rep.blockers[0].name


def test_deflated_sharpe_expected_max_grows_with_trials():
    a = sg.deflated_sharpe(1.0, 120, 2, 12.0)["expected_max"]
    b = sg.deflated_sharpe(1.0, 120, 50, 12.0)["expected_max"]
    assert 0 < a < b


# ---------------- ④ 回撤与极值 ----------------

def test_drawdown_low_false_positive():
    """第一版用「单期占累计比例」门槛 0.30，误报 27%。"""
    bad = 0
    N = 40
    for s in range(N):
        lvl, _ = _level(sg.check_drawdown, _noise(1000 + s))
        bad += lvl not in (OK, "SKIP")
    assert bad / N <= 0.15, f"误报率 {bad / N:.0%} 过高"


def test_drawdown_detects_outlier_period():
    """植入 6σ 单期 ⇒ 必须报离群点（可能是数据错误）。"""
    hit = 0
    N = 20
    for s in range(N):
        r = _noise(3000 + s)
        r.iloc[50] = r.mean() + 6 * r.std()
        lvl, _ = _level(sg.check_drawdown, r)
        hit += lvl not in (OK, "SKIP")
    assert hit / N >= 0.8, f"检出率只有 {hit / N:.0%}"


def test_drawdown_reports_max_dd_and_recovery():
    lvl, rep = _level(sg.check_drawdown, _noise(11))
    assert "max_drawdown" in rep.stats
    assert rep.stats["max_drawdown"] <= 0
    d = rep.findings[0].detail
    assert "最大回撤" in d and ("收复" in d or "未收复" in d)


def test_outlier_finding_does_not_overclaim():
    """★ 只能说「异常」，不能说「一定错」。"""
    r = _noise(3)
    r.iloc[50] = r.mean() + 8 * r.std()
    lvl, rep = _level(sg.check_drawdown, r)
    if lvl != OK:
        assert "不说它一定错" in rep.findings[0].impact


# ---------------- 输入形态 ----------------

def test_nav_input_converted_to_returns():
    r = _noise(11)
    nav = (1 + r).cumprod()
    back = sg.to_returns(nav, "nav")
    assert abs(float(back.iloc[-1]) - float(r.iloc[-1])) < 1e-12


def test_short_series_skipped_not_crashed():
    for fn in (sg.check_year_concentration, sg.check_nw_lag_sensitivity,
               sg.check_drawdown):
        rep = AuditReport()
        fn(pd.Series([0.01, 0.02], index=IDX[:2]), rep)
        assert rep.skipped, f"{fn.__name__} 没有跳过超短序列"
