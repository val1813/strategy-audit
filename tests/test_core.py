"""核心数学的对账型断言。

★ 为什么这个文件最重要
--------------------
实测教训：闸门自己也会标定错，连错四次之后才改用对账型断言。
所以这里不测「结果看起来合理」，只测【两条独立实现是否恒等】
和【已知答案能否被精确反推】。前者抓实现错误，后者抓口径错误。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from strategy_audit import core

from synth import equal_weight, make_prices, month_ends


# ---------------- 对账①：期间收益 vs 日连乘 ----------------

def test_period_returns_reconciles_with_daily_path(px, wm_clean):
    """区间内买入持有 ⇒ 两条独立实现必须恒等。

    period_returns 一次算清；daily_path 按日复利、权重随价格漂移。
    数学上恒等。偏差 > 1e-12 说明漂移权重的实现写错了 ——
    而写错之后净值仍然「看起来合理」，只有对账能抓到。
    """
    pr = core.period_returns(wm_clean, px, policy="hold_last")
    dp = core.daily_path(wm_clean, px)

    assert len(pr) > 10, "对账样本太少，测试本身无效"
    worst = 0.0
    for t0, row in pr.iterrows():
        seg = dp[(dp.index > t0) & (dp.index <= row["end"])]
        compounded = float((1.0 + seg).prod() - 1.0)
        worst = max(worst, abs(compounded - float(row["ret"])))
    assert worst < 1e-12, f"期间收益与日连乘不一致，最大偏差 {worst:.3e}"


def test_daily_path_covers_all_segments(px, wm_clean):
    """日收益序列应覆盖首末调仓日之间的每个交易日（不含起点）。"""
    dp = core.daily_path(wm_clean, px)
    lo, hi = wm_clean.index.min(), wm_clean.index.max()
    expected = [d for d in px.index if lo < d <= hi]
    assert len(dp) == len(expected), (
        f"日收益覆盖 {len(dp)} 天，应为 {len(expected)} 天 —— 有区间被跳过")


# ---------------- 对账②：单只标的退化为该标的自身收益 ----------------

def test_single_name_equals_its_own_return(px):
    """100% 押一只票 ⇒ 组合收益必须精确等于该票收益。

    这是最基本的口径检查。若归一或漂移写错，这里立刻不成立。
    """
    reb = month_ends(px)
    c = px.columns[0]
    wm = pd.DataFrame(0.0, index=pd.Index(reb, name="date"), columns=[c])
    wm[c] = 1.0
    pr = core.period_returns(wm, px[[c]])
    for t0, row in pr.iterrows():
        want = float(px.loc[row["end"], c] / px.loc[t0, c] - 1.0)
        assert abs(float(row["ret"]) - want) < 1e-14


# ---------------- 换手 ----------------

def test_turnover_zero_when_weights_never_change(px):
    """权重完全不变 ⇒ 漂移调整换手【不为 0】，朴素换手为 0。

    ★ 这一条把两个口径的差别钉死：
    目标权重不变，但价格动了会让实际持仓漂移，要拉回目标就必须交易。
    所以「不交易」的是朴素口径的假象，漂移口径才对。
    """
    reb = month_ends(px)
    codes = list(px.columns[:10])
    wm = pd.DataFrame(0.1, index=pd.Index(reb, name="date"), columns=codes)
    to = core.turnover(wm, px[codes])
    assert float(to["naive"].abs().max()) < 1e-15, "目标权重不变，朴素换手应为 0"
    assert float(to["drift_adj"].mean()) > 0, (
        "价格漂移后要拉回等权必须交易，漂移调整换手不应为 0")


def test_turnover_full_rotation_is_one(px):
    """完全换仓（无重叠持仓）⇒ 单边换手 = 100%。"""
    reb = month_ends(px)[:4]
    a, b = list(px.columns[:5]), list(px.columns[5:10])
    rows = []
    for i, t in enumerate(reb):
        for c in (a if i % 2 == 0 else b):
            rows.append((t, c, 0.2))
    wm = (pd.DataFrame(rows, columns=["date", "code", "weight"])
          .pivot(index="date", columns="code", values="weight")
          .fillna(0.0).sort_index())
    to = core.turnover(wm, px)
    assert np.allclose(to["naive"].values, 1.0, atol=1e-12)


def test_drift_weights_sum_to_one(px, wm_clean):
    """漂移后归一 ⇒ gross 必须回到 1。"""
    dates = list(wm_clean.index)
    for t0, t1 in zip(dates[:-1], dates[1:]):
        g = px.loc[t1] / px.loc[t0]
        w = wm_clean.loc[t0]
        d = core.drift_weights(w[w != 0], g)
        assert abs(float(d.abs().sum()) - 1.0) < 1e-12


# ---------------- 缺价政策 ----------------

@pytest.mark.parametrize("nc,nd,k,sd", [(20, 6, 8, 3), (30, 6, 8, 3),
                                        (40, 8, 10, 3), (25, 5, 6, 7),
                                        (30, 10, 8, 3), (50, 12, 10, 5)])
def test_missing_policies_ordered_when_delisting_crashes(nc, nd, k, sd):
    """退市前有崩盘时，三政策次序必须是 drop > hold_last > zero。

    ★ 这是「成分变动 ≠ 退市」那条铁律的可执行形式，但它【有前提】：
    次序由崩盘驱动 —— 无损移除之所以最高，是因为它扔掉了一段真实亏损。
    没有亏损可扔，就没有定向结论（见下一个测试）。
    6 组参数全过，才能说这个次序不是某个种子的巧合。
    """
    px_d = make_prices(n_codes=nc, n_dead=nd, seed=sd, death_drawdown=0.8)
    reb = month_ends(px_d)
    wm = equal_weight(px_d, reb, k=k, seed=2)
    tot = {p: float((1.0 + core.period_returns(wm, px_d, policy=p)["ret"]).prod())
           for p in core.MISSING_POLICIES}
    assert tot["drop"] > tot["hold_last"] > tot["zero"], tot


def test_missing_policy_order_is_not_universal():
    """★ 标的在中性价位凭空消失时，drop 与 hold_last 的高低会翻转。

    这一条是【反向断言】：它记录的是我原本搞错的事。
    第一版断言 drop ≥ hold_last 恒成立，只因为恰好选了成立的那个种子；
    实测 6 个面板里 3 个翻转。没有亏损可扔，drop 就只是
    「在存活标的上重新归一」，高低取决于存活标的当期涨跌 —— 没有定向意义。

    工具因此报【区间】而不报「哪种政策偏高」。
    """
    flips = 0
    for nc, nd, k, sd in [(20, 6, 8, 3), (30, 6, 8, 3), (40, 8, 10, 3),
                          (25, 5, 6, 7), (30, 10, 8, 3), (50, 12, 10, 5)]:
        px_d = make_prices(n_codes=nc, n_dead=nd, seed=sd, death_drawdown=0.0)
        wm = equal_weight(px_d, month_ends(px_d), k=k, seed=2)
        tot = {p: float((1.0 + core.period_returns(wm, px_d, policy=p)["ret"]).prod())
               for p in core.MISSING_POLICIES}
        # zero 始终最低（清算是真实损失），但 drop vs hold_last 无定向
        assert tot["zero"] < tot["hold_last"], tot
        if tot["drop"] < tot["hold_last"]:
            flips += 1
    assert flips > 0, "无崩盘时次序竟然全部成立 —— 反向断言失效，需重新核实"


def test_hold_last_keeps_the_pre_delisting_crash():
    """★ hold_last 必须保留退市前的崩盘，不能按买入价记 0%。

    这一条钉死一个实质 bug：第一版把缺价的毛增长设成 1.0，
    语义上是「按【买入】价持有」而非「按最后可见价格持有」——
    一只跌 80% 然后退市的票被记成 0% 收益，崩盘整段静默消失。
    """
    px_d = make_prices(n_codes=12, n_dead=1, seed=3, death_drawdown=0.8)
    dead = px_d.columns[0]
    reb = month_ends(px_d)
    wm = pd.DataFrame(0.0, index=pd.Index(reb, name="date"),
                      columns=list(px_d.columns))
    wm[dead] = 1.0                                  # 100% 押那只会退市的票
    pr = core.period_returns(wm, px_d, policy="hold_last")
    seg = pr[pr["n_missing"] > 0]
    assert len(seg) > 0, "构造未产生缺价事件"
    # 崩盘发生在退市前，所以首个缺价区间的收益必须显著为负
    assert float(seg["ret"].iloc[0]) < -0.20, (
        f"hold_last 丢掉了退市前的崩盘，记成 {float(seg['ret'].iloc[0]):+.4f}")


def test_period_returns_rejects_unknown_policy(px, wm_clean):
    with pytest.raises(ValueError, match="policy"):
        core.period_returns(wm_clean, px, policy="ffill")


def test_missing_price_counted(px):
    """缺价事件必须被计数并报出权重占比。"""
    px_d = make_prices(n_codes=20, n_dead=6, seed=3)
    reb = month_ends(px_d)
    wm = equal_weight(px_d, reb, k=8, seed=2)
    pr = core.period_returns(wm, px_d)
    assert int((pr["n_missing"] > 0).sum()) > 0
    assert float(pr["w_missing"].max()) > 0


# ---------------- 年化 ----------------

def test_annualize_recovers_known_rate():
    """已知恒定月收益 ⇒ 年化必须精确反推。"""
    r = pd.Series([0.01] * 36)
    a = core.annualize(r, 12.0)
    assert abs(a["ann_ret"] - (1.01 ** 12 - 1)) < 1e-12
    assert abs(a["sharpe"]) < 1e-9 or not np.isfinite(a["sharpe"])


def test_periods_per_year_monthly(px):
    """月频调仓 ⇒ 每年约 12 期。"""
    ppy = core.periods_per_year(month_ends(px))
    assert 11.0 < ppy < 13.5, ppy


@pytest.mark.parametrize("idx,want,tol", [
    (pd.bdate_range("2020-01-02", periods=11), 252.0, 12.0),
    (pd.date_range("2020-01-31", periods=11, freq="ME"), 12.0, 0.3),
    (pd.date_range("2020-01-03", periods=200, freq="W-FRI"), 52.2, 0.3),
    (pd.date_range("2020-01-01", periods=1000, freq="D"), 365.25, 0.1),
])
def test_periods_per_year_respects_observation_frequency(idx, want, tol):
    """日频交易序列扣周末；低频与 7x24 日频维持正确的原有口径。"""
    assert abs(core.periods_per_year(idx) - want) < tol
