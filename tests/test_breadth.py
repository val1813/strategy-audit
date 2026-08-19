"""族四：风险身份（残差有效注数）。

★ 这一族的度量必须先满足数学恒等，再谈解释
----------------------------------------
breadth = [w'diag(Σ)w] / [w'Σw] / Σwᵢ² 有几个可验证的定点：
残差真独立 ⇒ 等于 1/Σwᵢ²（等权即 n）；残差完全共动 ⇒ 趋于 1。
先钉死这些，再钉「对照必须同权重」这类识别检验的正确性。

★ 已经抓到的实质错误（回归钉在下面）
--------------------------------
同规模对照漏传权重 ⇒ 对照实际是等权、本账是市值加权，于是
「本账 7.0 注 vs 对照 30.7 注」里绝大部分差异来自加权方式而非选股。
修好后对照 12.2 注、结论从 WARN 变 OK。识别检验只允许变一个东西。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from strategy_audit import breadth as br
from strategy_audit.report import OK, WARN, AuditReport

from synth import equal_weight, make_prices, month_ends


# ---------------- 度量的数学定点 ----------------

def test_independent_residuals_give_n():
    """残差互不相关、等权 ⇒ 注数 = 标的数。"""
    rng = np.random.default_rng(1)
    n = 12
    resid = rng.normal(0, 0.02, (600, n))
    b = br.residual_breadth(resid)
    assert abs(b - n) / n < 0.15, b


def test_perfectly_comoving_residuals_give_one():
    """残差完全共动 ⇒ 注数趋于 1（只有一注赌）。"""
    rng = np.random.default_rng(2)
    common = rng.normal(0, 0.02, 600)
    resid = np.column_stack([common] * 10)
    b = br.residual_breadth(resid)
    assert b < 1.5, b


def test_breadth_never_exceeds_nominal_by_much():
    """★ 注数不该显著超过名义有效持仓数。

    超过就说明残差被机械压出了负相关 —— 共用一条含自己的市场均值
    就会这样（合成面板实测 20 只报 28.9 注）。留一代理是为了修这个。
    """
    rng = np.random.default_rng(3)
    n = 20
    resid = rng.normal(0, 0.02, (400, n))
    b = br.residual_breadth(resid)
    assert b <= n * 1.25, f"{b:.1f} 注 > {n} 只标的太多，疑似残差负相关"


def test_effective_names_equals_n_when_equal_weight():
    w = np.full(25, 1 / 25)
    assert abs(br.effective_names(w) - 25) < 1e-9


def test_effective_names_drops_with_concentration():
    w = np.array([0.5, 0.3, 0.1, 0.05, 0.05])
    assert br.effective_names(w) < 5


def test_single_name_returns_nan_not_crash():
    """一只票按定义就是一注，不该让 np.cov 塌成标量后崩掉。"""
    resid = np.random.default_rng(4).normal(0, 0.02, (100, 1))
    assert not np.isfinite(br.residual_breadth(resid))


def test_too_few_observations_returns_nan():
    resid = np.random.default_rng(5).normal(0, 0.02, (2, 8))
    assert not np.isfinite(br.residual_breadth(resid))


def test_enb_bounded_by_n():
    rng = np.random.default_rng(6)
    resid = rng.normal(0, 0.02, (500, 10))
    e = br.enb(resid)
    assert 0 < e <= 10 * 1.05, e


# ---------------- 识别检验：对照必须同权重 ----------------

def _panel_and_weights(n_codes=40, k=12, seed=7, tilt=False):
    px = make_prices(n_codes=n_codes, seed=seed)
    reb = month_ends(px)
    if not tilt:
        return px, equal_weight(px, reb, k=k, seed=2)
    # 集中加权：权重差异很大，等权/加权的注数会明显不同
    rows = []
    rng = np.random.default_rng(seed)
    for t in reb:
        pool = [c for c in px.columns if np.isfinite(px.loc[t, c])]
        if len(pool) < k:
            continue
        pick = list(rng.choice(pool, k, replace=False))
        ramp = np.linspace(0.30, 0.02, k)
        ramp = ramp / ramp.sum()
        for c, w in zip(pick, ramp):
            rows.append((t, c, float(w)))
    m = (pd.DataFrame(rows, columns=["date", "code", "weight"])
         .pivot(index="date", columns="code", values="weight")
         .fillna(0.0).sort_index())
    return px, m.reindex(sorted(m.columns), axis=1)


def test_control_uses_same_weights_as_the_book():
    """★ 回归：对照与本账必须用同一个权重向量。

    漏传 w 时对照是等权、本账是集中加权，比值会被加权方式污染。
    检验方式：集中加权的组合下，若对照仍是等权，则 ctrl 会明显高于
    该组合自身的等权注数上限（≈持仓数），从而暴露口径不一致。
    """
    px, wm = _panel_and_weights(tilt=True)
    d, notes = br.residual_breadth_panel(wm, px)
    if not len(d) or d["breadth_ctrl"].dropna().empty:
        pytest.skip("该合成面板未产生可用对照")
    ne = float(d["ne_nominal"].median())
    ctrl = float(d["breadth_ctrl"].median())
    # 对照用了同样的集中权重 ⇒ 其注数上限也是 ne（而非持仓只数）
    assert ctrl <= ne * 1.3, (
        f"对照 {ctrl:.1f} 注远超本账名义有效持仓数 {ne:.1f} "
        "—— 对照很可能用了等权而本账是加权")


def test_control_ratio_near_one_for_random_picks():
    """随机选股的组合，本账注数应与同规模随机对照相当（比值≈1）。"""
    px, wm = _panel_and_weights()
    d, _ = br.residual_breadth_panel(wm, px)
    if not len(d) or d["breadth_ctrl"].dropna().empty:
        pytest.skip("该合成面板未产生可用对照")
    rep = AuditReport()
    br.check_breadth_control(d, rep)
    r = rep.stats.get("breadth_control_ratio")
    assert r is None or 0.5 <= r <= 1.8, f"随机选股的比值 {r} 偏离 1 太多"


def test_control_skipped_when_universe_too_small():
    """股票池不足持仓数 2 倍 ⇒ 对照会与本账大量重叠，必须跳过并说明。"""
    px, wm = _panel_and_weights(n_codes=14, k=10)
    d, _ = br.residual_breadth_panel(wm, px)
    rep = AuditReport()
    br.check_breadth_control(d, rep)
    if d is None or not len(d) or d["breadth_ctrl"].dropna().empty:
        assert rep.skipped and "2 倍" in rep.skipped[0]


# ---------------- 报告行为 ----------------

def test_family_never_emits_block():
    """★ 这一族最高只到 WARN —— 注数压缩不让净值算错，只让风险预算算错。"""
    px, wm = _panel_and_weights(tilt=True)
    d, notes = br.residual_breadth_panel(wm, px)
    rep = AuditReport()
    br.check_residual_breadth(d, rep, notes)
    br.check_breadth_control(d, rep)
    br.check_breadth_vs_enb(d, rep)
    assert not rep.blockers, [f.name for f in rep.blockers]


def test_sampling_is_disclosed():
    """★ 抽样必须写进报告 —— 静默截断会被读成「全都测过了」。"""
    px, wm = _panel_and_weights(n_codes=30, k=8)
    d, notes = br.residual_breadth_panel(wm, px, max_dates=5)
    if len(d) and len(wm.index) > 5:
        assert any("取 1 个" in n or "抽样" in n for n in notes), notes


def test_deterministic_across_runs():
    """同一份输入必须给同一份报告（对照抽样用固定种子）。"""
    px, wm = _panel_and_weights()
    a, _ = br.residual_breadth_panel(wm, px)
    b, _ = br.residual_breadth_panel(wm, px)
    if len(a) and len(b):
        pd.testing.assert_frame_equal(a, b)


def test_skips_cleanly_when_too_few_names():
    """持仓少于阈值 ⇒ 整族跳过，不能报出无意义的注数。"""
    px = make_prices(n_codes=10, seed=3)
    reb = month_ends(px)
    wm = equal_weight(px, reb, k=3, seed=2)
    d, notes = br.residual_breadth_panel(wm, px)
    rep = AuditReport()
    br.check_residual_breadth(d, rep, notes)
    assert rep.skipped or not rep.findings


def test_report_states_weight_sensitivity():
    """加权与等权注数差很多时，报告必须点出「结论对权重敏感」。"""
    px, wm = _panel_and_weights(tilt=True)
    d, notes = br.residual_breadth_panel(wm, px)
    if not len(d):
        pytest.skip("无可用结果")
    rep = AuditReport()
    br.check_residual_breadth(d, rep, notes)
    b = rep.stats.get("breadth_residual")
    bew = rep.stats.get("breadth_residual_ew")
    if b and bew and abs(bew - b) / b > 0.2:
        assert "等权" in rep.findings[0].detail


def test_control_verdict_matches_its_own_numbers():
    """★ 回归：结论不许和自己给出的数字相反。

    第一版拿 b/c < 0.5 当门槛，于是真面板上「本账 6.5 注 vs 对照 11.8 注」
    （比值 0.55）被判 OK 并写成「不是这本账的选股特征」—— 6.5 相对 11.8
    已接近腰斩。0.55 与 0.49 之间没有实质差别，却给出完全相反的对外说法。
    现在按【本账落在对照抽样分布的哪个分位】判。
    """
    px, wm = _panel_and_weights(tilt=True)
    d, _ = br.residual_breadth_panel(wm, px)
    if not len(d) or d["breadth_ctrl"].dropna().empty:
        pytest.skip("该合成面板未产生可用对照")
    rep = AuditReport()
    br.check_breadth_control(d, rep)
    f = [x for x in rep.findings if "对照" in x.name][0]
    b = float(d["breadth"].median())
    c = float(d["breadth_ctrl"].median())
    pct = rep.stats.get("breadth_control_pctile")
    # 本账明显低于对照（腰斩级）时不许说「不是选股特征」
    if b < 0.7 * c:
        assert f.level == WARN, (
            f"本账 {b:.1f} 注 vs 对照 {c:.1f} 注 却判 {f.level}")
        assert "不是这本账的选股特征" not in f.impact
    # 分位必须出现在报告里，让用户看到判据本身
    if pct is not None and np.isfinite(pct):
        assert "分位" in f.detail or "%" in f.detail


def test_control_percentile_recorded_per_date():
    """分位是逐调仓日算的，必须落进结果表以便复核。"""
    px, wm = _panel_and_weights()
    d, _ = br.residual_breadth_panel(wm, px)
    if len(d) and not d["breadth_ctrl"].dropna().empty:
        assert "ctrl_pctile" in d.columns
        v = d["ctrl_pctile"].dropna()
        assert len(v) and v.between(0, 1).all()


# ---------------- 报告自洽性（四处措辞 bug 的回归） ----------------
#
# ★ 这四个都是【跑起来才看见】的，测试套件当时全绿。
# 合成面板 + demo + 真实面板各抓到两个，共同点是：数字算对了，
# 但报告把它叙述成了自相矛盾的话。审计工具的输出就是产品，
# 措辞错误等于结论错误。

def _mk(n, held_extra=0.0, k=20, seed=5, f_sd=0.015, e_sd=0.015):
    """按需造面板：held_extra>0 时前 k 只额外共享一个因子（真压缩）。"""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2021-01-04", "2023-12-29")
    R = rng.normal(0, e_sd, (len(dates), n))
    R += np.outer(rng.normal(0, f_sd, len(dates)), np.ones(n))
    if held_extra > 0:
        R[:, :k] += np.outer(rng.normal(0, held_extra, len(dates)), np.ones(k))
    px = pd.DataFrame(100 * np.exp(np.cumsum(R, axis=0)), index=dates,
                      columns=[f"{i:06d}.SZ" for i in range(1, n + 1)])
    rows = [(t, c, 1.0 / k) for t in month_ends(px) for c in px.columns[:k]]
    wm = (pd.DataFrame(rows, columns=["date", "code", "weight"])
          .pivot(index="date", columns="code", values="weight")
          .fillna(0.0).sort_index())
    return px, wm


def _report(px, wm):
    d, notes = br.residual_breadth_panel(wm, px)
    rep = AuditReport()
    br.check_residual_breadth(d, rep, notes)
    br.check_breadth_control(d, rep)
    return d, rep


def test_no_understatement_claim_when_ratio_is_noise():
    """★ ratio≈1 时不许写「低估 1.01 倍」。

    实测两次翻车：demo 报「低估 0.98 倍」（比值 <1，语义上根本没低估）、
    真实面板随机组合报「低估 1.01 倍」（纯估计噪声被叙述成低估）。
    """
    px, wm = _mk(260)                     # 干净大池 ⇒ 注数≈持仓数
    _, rep = _report(px, wm)
    f = [x for x in rep.findings if x.name == "残差有效注数"][0]
    ne, b = rep.stats["breadth_ne_nominal"], rep.stats["breadth_residual"]
    if ne / b <= 1.0 + br.UNDERSTATE_NOISE:
        assert "低估" not in f.impact, f.impact
        assert "无需打折" in f.impact
        assert f.level == OK


def test_debiased_breadth_never_exceeds_nominal():
    """★ 去偏值也受 1/Σw² 硬上界约束。

    去偏是一阶近似，小 complement 且 b 已接近 ne 时会反解出超过持仓数的
    注数（demo 实测 8 只报 12.9 注）—— 报出去就是自相矛盾。
    """
    for n in (30, 60, 120):
        px, wm = _mk(n)
        _, rep = _report(px, wm)
        adj = rep.stats.get("breadth_proxy_adjusted")
        ne = rep.stats["breadth_ne_nominal"]
        if adj is not None and np.isfinite(adj):
            assert adj <= ne + 1e-9, f"池 {n}: 去偏 {adj:.1f} > 名义 {ne:.1f}"


def test_full_percentile_is_not_worded_as_a_percentage():
    """★ pct=1.0 不许写成「低于 100% 的同规模随机组合」。

    真实面板上低波组合就是 pct=1.0，那句话读起来像「比所有组合都低、
    包括它自己」。20 次抽样只能分辨到 1/20，所以要报成「全部 20 次」。
    """
    px, wm = _mk(260, held_extra=0.02)    # 真压缩 ⇒ 分位顶到 1.0
    _, rep = _report(px, wm)
    pct = rep.stats.get("breadth_control_pctile")
    if pct is not None and pct >= 1.0 - 1e-9:
        f = [x for x in rep.findings if x.name == "同规模对照"][0]
        assert "100%" not in f.detail and "100%" not in f.impact
        assert f"全部 {br.N_CTRL_DRAWS} 次" in f.detail


def test_two_findings_never_contradict_each_other():
    """★ ①说「无需打折」时②不许说「低估倍数仍然成立」。

    两项必须用同一条判据（ne/b > 1+UNDERSTATE_NOISE）。②原先只判 >=1.0，
    于是纯噪声也会让那句话出现。
    """
    for n, extra in ((260, 0.0), (60, 0.0), (260, 0.02)):
        px, wm = _mk(n, held_extra=extra)
        _, rep = _report(px, wm)
        f1 = [x for x in rep.findings if x.name == "残差有效注数"][0]
        f2 = [x for x in rep.findings if x.name == "同规模对照"][0]
        if "无需打折" in f1.impact:
            assert "仍然成立" not in f2.impact, (n, extra, f2.impact)
