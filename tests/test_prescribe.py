"""族六：处方层。

★ 这一族的测试比别的族更重要，因为它是唯一【给建议】的一族。
诊断给错了，用户去查一个不存在的问题；处方给错了，用户按噪声改策略。
所以这里的核心断言不是「能不能给出建议」，而是：

    · 恒等式：拆分之和 = 总换手（算术保证，不是实证）
    · 身份钉住：名单重合恒 100%、τ(φ) 单调（算术保证）
    · 阴性对照：名单换血为主的组合必须报【不可处方】
    · 阳性对照：微调为主的组合必须认出旋钮存在
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from strategy_audit import core
from strategy_audit import prescribe as pr
from strategy_audit.report import AuditReport

# ★ 平铺 import：conftest 已把 tests/ 加进 sys.path（与其它测试一致）。
# 不用 `prescribe as px` —— px 是 conftest 里的价格面板 fixture 名，
# 撞名会让「模块」和「面板」在测试里混成一个符号。
from synth import equal_weight, make_prices, month_ends, tilted_weight


@pytest.fixture(scope="module")
def panel():
    p = make_prices(n_codes=60, seed=5)
    return p, month_ends(p)


# ---------------- 恒等式：算术保证，不是实证 ----------------

def test_split_sums_to_drift_adjusted_turnover(panel):
    """★ 名单换血 + 权重微调 必须恒等于漂移调整口径的总换手。

    这不是「大致相等」——两部分是同一个 0.5·Σ|Δw| 在互斥标的集合上的
    划分，所以必须精确相等。不等就是拆分漏了一类标的。
    """
    p, reb = panel
    wm = equal_weight(p, reb, k=12)
    s = pr.split_turnover(wm, p)
    to = core.turnover(wm, p)
    # split 用「上期→本期」的漂移，与 core.turnover 同口径
    assert s["churn"] + s["tweak"] == pytest.approx(
        float(to["drift_adj"].mean()), rel=1e-9), (s, to["drift_adj"].mean())


def test_split_share_in_unit_interval(panel):
    p, reb = panel
    s = pr.split_turnover(tilted_weight(p, reb, k=12), p)
    assert 0.0 <= s["share"] <= 1.0


# ---------------- 身份钉住：算术保证 ----------------

@pytest.mark.parametrize("phi", [1.0, 0.6, 0.2, 0.0])
def test_identity_controlled_keeps_names(panel, phi):
    """★ 名单重合必须恒为 100%，与 φ 无关。

    第一版按 |Δw| 排序削预算（不分名单变动与微调），实测 φ=0.4 时
    与目标名单重合仅 1%、最大权重 95.7%、有效注数 1.5 —— 那个
    「+21% 毛收益改善」是一只股票，不是省下的成本。
    名单变动无条件全执行就是为了钉死这一点。
    """
    p, reb = panel
    wm = tilted_weight(p, reb, k=12)
    r = pr.identity_controlled_path(wm, p, phi)
    assert r["overlap"] == pytest.approx(1.0, abs=1e-9), phi


def test_tau_monotone_in_phi(panel):
    """★ τ(φ) 必须单调不增 —— 算术保证，不是实证发现。

    不单调会让「c* 最低的那个 φ」变成在噪声里挑最小值。
    实测：阈值带宽版本 b=3% 给 τ=61%、b=5% 给 τ=87%，因此作废。
    """
    p, reb = panel
    wm = tilted_weight(p, reb, k=12)
    taus = [float(pr.identity_controlled_path(wm, p, f)["tau"].mean())
            for f in pr.PHI_GRID]
    assert all(a >= b - 1e-12 for a, b in zip(taus, taus[1:])), taus


def test_phi_one_equals_target_weights(panel):
    """φ=1 必须与「完全按目标调仓」逐期恒等（否则基准就错了）。"""
    p, reb = panel
    wm = equal_weight(p, reb, k=12)
    got = pr.identity_controlled_path(wm, p, 1.0)["ret"]
    want = core.period_returns(wm, p)["ret"]
    n = min(len(got), len(want))
    assert np.allclose(got.values[:n], want.values[:n], atol=1e-10)


@pytest.mark.parametrize("w,tag", [
    ([1 / 6] * 6, "长仓等权 gross=1"),
    ([.3, .25, .2, .1, .1, .05], "长仓不等权 gross=1"),
    ([.5, .5, 0, 0, -.5, -.5], "市场中性 Σw=0 gross=2"),
    ([.2, .2, .2, .1, .05, .05], "留现金 Σw=0.8"),
    ([1 / 3] * 6, "2倍杠杆 Σw=2"),
])
def test_reconciles_under_any_gross_convention(w, tag):
    """★ 对账型断言：φ=1 必须等于 core 的独立实现，与 gross 口径无关。

    这条抓到过一个实质 bug。core.period_returns 【不】归一（收益随 gross
    线性缩放），而本模块按单位资本算。两者在 gross=1 时恒等 —— 而
    audit() 上游已归一，所以走公开 API 完全看不出来。直接调用本模块时：

        市场中性 Σw=0、gross=2   差 0.5×
        留了现金 Σw=0.8          差 1.25×
        2 倍杠杆 Σw=2            差 0.5×

    交叉成本 c* 是个 bp 量，差 2 倍就是把结论说错。修法是在本模块每个
    入口先 _norm_gross()。这正是 README 那句「凡是能用两条独立路径算的量
    就都算一遍对账」——闸门自己也会标定错，只有对账能抓到。
    """
    from strategy_audit.contract import normalize_gross
    idx = pd.bdate_range("2021-01-04", periods=60)
    rng = np.random.default_rng(4)
    p = pd.DataFrame({c: 100 * np.exp(np.cumsum(rng.normal(0, .02, 60)))
                      for c in list("abcdef")}, index=idx)
    reb = idx[::10]
    wm = pd.DataFrame([w] * len(reb), index=reb, columns=list("abcdef"))
    got = pr.identity_controlled_path(wm, p, 1.0)["ret"]
    want = core.period_returns(normalize_gross(wm), p)["ret"]
    n = min(len(got), len(want))
    assert np.allclose(got.values[:n], want.values[:n], atol=1e-10), tag


def test_market_neutral_does_not_crash():
    """★ Σw=0 的市场中性组合：不许崩，也不许报出 nan。

    gross 归一的分母是 |w| 之和（=2），不是 Σw（=0）—— 若用后者
    就会全表除零。这类组合在机构里很常见，不能只支持长仓。
    """
    idx = pd.bdate_range("2021-01-04", periods=60)
    rng = np.random.default_rng(9)
    p = pd.DataFrame({c: 100 * np.exp(np.cumsum(rng.normal(0, .02, 60)))
                      for c in list("abcdef")}, index=idx)
    reb = idx[::10]
    wm = pd.DataFrame([[.5, .5, 0, 0, -.5, -.5]] * len(reb),
                      index=reb, columns=list("abcdef"))
    rep = AuditReport()
    pr.check_turnover_value(wm, p, rep)
    pr.check_turnover_split(wm, p, rep)
    pr.check_prescription(wm, p, 12.0, rep)
    assert not rep.blockers
    assert "nan" not in rep.text().lower()


# ---------------- 换手的边际价值 ----------------

def test_turnover_value_zero_when_no_trading(panel):
    """权重从不变动 ⇒ 没有可评价的调仓，必须跳过而不是报 0bp。"""
    p, reb = panel
    codes = list(p.columns[:10])
    wm = pd.DataFrame(0.1, index=pd.DatetimeIndex(reb), columns=codes)
    wm = wm.reindex(columns=sorted(p.columns), fill_value=0.0)
    rep = AuditReport()
    pr.check_turnover_value(wm, p, rep)
    assert any("换手的边际价值" in s for s in rep.skipped) or rep.findings


def test_turnover_value_positive_on_planted_skill(panel):
    """★ 阳性对照：换手确实买到东西的组合必须报 OK 而不是 WARN。

    构造：每期把权重全部压到【下期】涨得最多的标的上（真前视）——
    这种换手在毛口径上必然强正，若报成「负贡献」就是符号错了。
    """
    p, reb = panel
    rows = []
    for t0, t1 in zip(reb[:-1], reb[1:]):
        fwd = (p.loc[t1] / p.loc[t0] - 1.0).dropna().sort_values(ascending=False)
        for c in fwd.index[:8]:
            rows.append((t0, c, 1.0 / 8))
    wm = (pd.DataFrame(rows, columns=["date", "code", "weight"])
          .pivot(index="date", columns="code", values="weight")
          .fillna(0.0).sort_index())
    wm = wm.reindex(sorted(set(wm.columns) | set(p.columns)),
                    axis=1, fill_value=0.0)
    rep = AuditReport()
    d = pr.check_turnover_value(wm, p, rep)
    assert len(d) >= 6
    assert rep.stats["turnover_value_bp"] > 0, rep.stats
    assert not rep.blockers


def test_turnover_value_reports_mde(panel):
    """★ 负结果必须附最小可检出效应。

    「没查出问题」和「查不出这么小的问题」是两回事 —— 这条是全库的规矩。
    """
    p, reb = panel
    rep = AuditReport()
    pr.check_turnover_value(equal_weight(p, reb, k=12), p, rep)
    assert "turnover_value_mde_bp" in rep.stats
    txt = " ".join(f.detail for f in rep.findings)
    assert "最小可检出" in txt


# ---------------- 处方闸门 ----------------

def test_refuses_when_churn_dominates(panel):
    """★ 阴性对照：名单换血为主 ⇒ 必须报【不可处方】。

    实测（真实 A 股 800 只 / 月频 30 只反转）：微调仅占换手 1%，
    名单换血 95% —— 削微调省不下任何东西。此时给建议就是骗人。
    等权随机换名的合成组合同属这一类。
    """
    p, reb = panel
    wm = equal_weight(p, reb, k=12)      # 每期重新随机抽名字 ⇒ 全是换血
    rep = AuditReport()
    s = pr.split_turnover(wm, p)
    assert s["share"] < pr.TWEAK_SHARE_FLOOR, s
    pr.check_prescription(wm, p, core.periods_per_year(wm.index), rep)
    names = [f.name for f in rep.findings]
    assert any("不可处方" in n for n in names), names
    # ★ 拒绝给建议 ≠ 查过通过：必须说明这是拒绝
    txt = " ".join(f.impact for f in rep.findings)
    assert "拒绝给出建议" in txt


def test_recognises_knob_when_tweaks_dominate(panel):
    """★ 阳性对照：名单稳定、只有权重漂移 ⇒ 必须认出旋钮存在。

    构造：固定 12 只名字，目标权重每期都是同一组不等权 —— 名单零换血，
    换手全部来自价格漂移后拉回目标（100% 微调）。
    这正是市值加权大盘组合的形态（真实面板实测微调占 73%）。
    """
    p, reb = panel
    codes = sorted(p.columns[p.loc[reb[-1]].notna()])[:12]
    w = np.linspace(0.16, 0.02, len(codes))
    w = w / w.sum()
    wm = pd.DataFrame([w] * len(reb), index=pd.DatetimeIndex(reb),
                      columns=codes)
    wm = wm.reindex(columns=sorted(p.columns), fill_value=0.0)
    s = pr.split_turnover(wm, p)
    assert s["churn"] == pytest.approx(0.0, abs=1e-12), s
    assert s["share"] == pytest.approx(1.0, rel=1e-9), s
    rep = AuditReport()
    pr.check_prescription(wm, p, core.periods_per_year(wm.index), rep)
    names = [f.name for f in rep.findings]
    assert not any("没有可削的换手预算" in n for n in names), names


def test_prescription_never_blocks(panel):
    """★ 族六不许出 BLOCK。

    BLOCK 的语义是「净值不可信，先修」。处方层不质疑净值 ——
    它说的是「有个不需要预测的改动能让净收益更好」。
    把优化建议做成 BLOCK 会逼用户为了过 CI 去改策略。
    """
    p, reb = panel
    for wm in (equal_weight(p, reb, k=12), tilted_weight(p, reb, k=12)):
        rep = AuditReport()
        ppy = core.periods_per_year(wm.index)
        pr.check_turnover_value(wm, p, rep)
        pr.check_turnover_split(wm, p, rep)
        pr.check_prescription(wm, p, ppy, rep)
        assert not rep.blockers, [f.name for f in rep.blockers]


def test_crossover_cost_needs_no_cost_assumption(panel):
    """c* 只能由两条毛曲线算出，不许依赖任何费率入参。"""
    import inspect
    sig = inspect.signature(pr.crossover_cost)
    bad = [p_ for p_ in sig.parameters if "cost" in p_ or "bp" in p_]
    assert not bad, bad


def test_floor_sits_in_an_empty_band(panel):
    """★ 门槛必须落在【空档】里，不能落在形态分布的密集处。

    真实 A 股面板上扫 12 种组合形态（持仓数 30/100/200 × 市值/等权 ×
    固定名单/反转换名）实测微调占比：

        反转换名的六种   1%、1%、1%、2%、3%、7%    ← 削了只省 1% 换手
        固定名单的六种   88%、89%、90%、93%、93%、95% ← 削了省 86% 换手

    7% 与 88% 之间没有任何形态 —— 门槛定在这个空档里，所以它对
    「削得下 / 削不下」的判别不依赖门槛的具体取值。
    合成面板复现同一个二分：随机换名 ≈ 0%，固定名单 = 100%。
    """
    p, reb = panel
    churny = pr.split_turnover(equal_weight(p, reb, k=12), p)["share"]

    codes = sorted(p.columns[p.loc[reb[-1]].notna()])[:12]
    w = np.linspace(0.16, 0.02, len(codes))
    wm = pd.DataFrame([w / w.sum()] * len(reb),
                      index=pd.DatetimeIndex(reb), columns=codes)
    tweaky = pr.split_turnover(
        wm.reindex(columns=sorted(p.columns), fill_value=0.0), p)["share"]

    assert churny < pr.TWEAK_SHARE_FLOOR <= tweaky, (churny, tweaky)
    # 空档要足够宽：门槛挪动 ±10pp 不该改变任何一边的裁决
    assert churny < pr.TWEAK_SHARE_FLOOR - 0.10
    assert tweaky > pr.TWEAK_SHARE_FLOOR + 0.10


def test_refusal_is_justified_by_savings(panel):
    """★ 拒绝处方必须【有理由】：削到 φ=0 确实省不下换手。

    这是闸门 A 的真正判据。若某个形态微调占比低但削了能省很多换手，
    那门槛就是错的 —— 实测门槛之下六种形态平均只省 1%。
    """
    p, reb = panel
    wm = equal_weight(p, reb, k=12)
    assert pr.split_turnover(wm, p)["share"] < pr.TWEAK_SHARE_FLOOR
    t1 = float(pr.identity_controlled_path(wm, p, 1.0)["tau"].mean())
    t0 = float(pr.identity_controlled_path(wm, p, 0.0)["tau"].mean())
    saved = (t1 - t0) / max(t1, 1e-12)
    assert saved < 0.15, f"削到 φ=0 省了 {saved:.0%}，门槛可能定错了"


def test_grid_is_predeclared():
    """★ φ 网格必须是模块常量，不能按结果动态加密。

    「扫一个更细的网格找更好的点」正是过拟合本身。
    """
    assert pr.PHI_GRID[0] == 1.0            # 第一档必须是「当前做法」
    assert pr.PHI_GRID[-1] == 0.0
    assert list(pr.PHI_GRID) == sorted(pr.PHI_GRID, reverse=True)


# ---------------- 敌意输入 ----------------

def test_short_panel_skips_cleanly(panel):
    """期数不足时给明确原因，不许崩也不许交空结论。"""
    p, reb = panel
    wm = equal_weight(p, reb[:3], k=8)
    rep = AuditReport()
    pr.check_turnover_value(wm, p, rep)
    pr.check_turnover_split(wm, p, rep)
    pr.check_prescription(wm, p, 12.0, rep)
    assert rep.skipped or rep.findings
    assert not rep.blockers
