"""族五：容量与可成交性。

★ 这一族之前【没有测试文件】—— 它是随族六一起补上的。
两个 A 股实质缺陷都是补测试时抓出来的：

  ① 板块判定只认 `sh.`/`sz.` 前缀 ⇒ `300750.SZ`、裸 6 位这些同样常见的
    写法全部落到默认 10%，创业板/科创板的限幅判小一半
  ② 涨跌停用 |r| ≥ 限幅 一律算挡住 ⇒ 「涨停要卖」「跌停要买」这两类
    【能正常成交】的交易被误判，不可成交权重高估 160%
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from strategy_audit import capacity as cp
from strategy_audit.report import AuditReport

from synth import make_prices, month_ends


# ---------------- ① 板块限幅判定 ----------------

@pytest.mark.parametrize("code,want", [
    # 带交易所前缀（第一版只认这一种）
    ("sh.600000", 0.10), ("sz.000001", 0.10), ("sz.300750", 0.20),
    ("sh.688981", 0.20), ("bj.430047", 0.30),
    # ★ 后缀写法 —— 同样常见，第一版全判成 10%
    ("600000.SH", 0.10), ("300750.SZ", 0.20), ("688981.SH", 0.20),
    ("430047.BJ", 0.30), ("873223.BJ", 0.30),
    # ★ 裸 6 位 —— 第一版全判成 10%
    ("600000", 0.10), ("300750", 0.20), ("688981", 0.20), ("920001", 0.30),
    # 其它号段
    ("sz.301001", 0.20), ("sh.605001", 0.10), ("sz.002415", 0.10),
    ("689009.SH", 0.20), ("SH600000", 0.10),
])
def test_board_limit_covers_all_common_code_formats(code, want):
    """★ 限幅必须按【6 位数字代码】判，不能按交易所前缀判。

    限幅判小了会把正常涨跌当成触板（20% 限幅的票涨 12% 被报成涨停），
    于是创业板/科创板的不可成交比例被系统性高估。
    """
    assert cp._board_limit(code) == pytest.approx(want), code


def test_board_limit_unknown_code_falls_back_not_crash():
    """认不出来的代码给默认限幅，不许崩 —— 用户的代码格式千奇百怪。"""
    for c in ("", "ABC", "AAPL", "000001.HK", None, 123456):
        assert 0.0 < cp._board_limit(c) <= 0.30


# ---------------- ② 涨跌停必须区分方向 ----------------

def _one_limit_panel(direction: str, side: str):
    """造一个只有单只标的触板的两期面板。

    direction  "up" 涨停 / "down" 跌停
    side       "buy" 该日要买入 / "sell" 该日要卖出
    """
    dates = pd.bdate_range("2021-01-04", periods=3)
    codes = ["sz.000001", "sz.000002"]
    px = pd.DataFrame(1.0, index=dates, columns=codes)
    # 第 2 天让 000001 触板（主板 ±10%）
    px.loc[dates[1], "sz.000001"] = 1.11 if direction == "up" else 0.89
    px.loc[dates[2]] = px.loc[dates[1]]
    # 权重：第 2 期对 000001 加仓（买）或减仓（卖）
    w0 = {"sz.000001": 0.5, "sz.000002": 0.5}
    w1 = ({"sz.000001": 0.9, "sz.000002": 0.1} if side == "buy"
          else {"sz.000001": 0.1, "sz.000002": 0.9})
    wm = pd.DataFrame([w0, w1], index=[dates[0], dates[1]])[codes]
    return wm, px


@pytest.mark.parametrize("direction,side,blocked", [
    ("up", "buy", True),      # 涨停要买入 ⇒ 真的买不进
    ("down", "sell", True),   # 跌停要卖出 ⇒ 真的卖不出
    ("up", "sell", False),    # ★ 涨停要卖出 ⇒ 卖得掉，不该算挡住
    ("down", "buy", False),   # ★ 跌停要买入 ⇒ 买得到，不该算挡住
])
def test_limit_blocks_only_the_matching_direction(direction, side, blocked):
    """★ 涨停挡买入、跌停挡卖出 —— 挡住的是相反的操作。

    第一版用 |r| ≥ 限幅 一律算不可成交。盲测实测 77 个触板标的-期里
    有 69 个属于「涨停要卖」「跌停要买」这两类能正常成交的交易，
    不可成交权重被高估 160%（1.30% vs 真实 0.50%）。
    高估 2.6 倍不叫保守，那是把能执行的策略报成执行不了。
    """
    wm, px = _one_limit_panel(direction, side)
    d = cp.untradable_weight(wm, px)
    assert len(d) == 1, d
    row = d.iloc[0]
    if blocked:
        assert row["n_limit"] == 1 and row["w_blocked"] > 0, row
        assert row["n_limit_ok"] == 0
    else:
        assert row["n_limit"] == 0 and row["w_blocked"] == 0, row
        # 触板但方向不挡的必须【被记下来】，不能悄悄消失
        assert row["n_limit_ok"] == 1, row


def test_released_limit_events_are_reported_not_hidden():
    """★ 被方向判定放行的触板必须出现在报告里。

    「放行了 69 笔」是审计结论的一部分 —— 静默放行等于用户无从判断
    这个推断是否合理。
    """
    wm, px = _one_limit_panel("up", "sell")
    rep = AuditReport()
    cp.check_untradable(wm, px, rep)
    txt = " ".join(f.detail for f in rep.findings)
    assert "方向不挡" in txt
    assert rep.stats["untradable_n_limit_passed"] == 1


def test_no_price_blocks_both_directions():
    """无价格（疑似停牌）两个方向都挡 —— 这一条与涨跌停不同。"""
    dates = pd.bdate_range("2021-01-04", periods=3)
    codes = ["sz.000001", "sz.000002"]
    px = pd.DataFrame(1.0, index=dates, columns=codes)
    px.loc[dates[1], "sz.000001"] = np.nan
    for side in ("buy", "sell"):
        w1 = ({"sz.000001": 0.9, "sz.000002": 0.1} if side == "buy"
              else {"sz.000001": 0.1, "sz.000002": 0.9})
        wm = pd.DataFrame([{"sz.000001": .5, "sz.000002": .5}, w1],
                          index=[dates[0], dates[1]])[codes]
        d = cp.untradable_weight(wm, px)
        assert d.iloc[0]["n_nopx"] == 1, side
        assert d.iloc[0]["w_blocked"] > 0, side


def test_st_flag_tightens_limit_to_five_percent():
    """★ ST 股限幅 5%：传了标志位就该按 5% 判，不传则按板块 10%。

    没有标志位时 ST 股的 ±5% 触板会被当成没触板，方向是【低估】
    不可成交 —— 与其它误判方向相反，所以必须能被显式修正。
    """
    dates = pd.bdate_range("2021-01-04", periods=3)
    codes = ["sz.000001", "sz.000002"]
    px = pd.DataFrame(1.0, index=dates, columns=codes)
    px.loc[dates[1], "sz.000001"] = 1.052        # +5.2%：ST 触板，主板不触
    px.loc[dates[2]] = px.loc[dates[1]]
    wm = pd.DataFrame([{"sz.000001": .5, "sz.000002": .5},
                       {"sz.000001": .9, "sz.000002": .1}],
                      index=[dates[0], dates[1]])[codes]

    assert cp.untradable_weight(wm, px).iloc[0]["n_limit"] == 0
    st = pd.DataFrame(False, index=dates, columns=codes)
    st.loc[:, "sz.000001"] = True
    assert cp.untradable_weight(wm, px, st=st).iloc[0]["n_limit"] == 1


# ---------------- ③ 容量与不出 BLOCK ----------------

@pytest.fixture(scope="module")
def panel():
    p = make_prices(n_codes=30, seed=13)
    return p, month_ends(p)


def _equal_w(px, reb, k=10):
    codes = sorted(px.columns[px.loc[reb[-1]].notna()])[:k]
    wm = pd.DataFrame(1.0 / len(codes), index=pd.DatetimeIndex(reb),
                      columns=codes)
    return wm.reindex(columns=sorted(px.columns), fill_value=0.0)


def test_capacity_scales_linearly_with_participation(panel):
    """容量对参与率线性 —— 报告这么写的，就必须真的是这样。"""
    p, reb = panel
    wm = _equal_w(p, reb)
    am = pd.DataFrame(1e8, index=p.index, columns=p.columns)
    a = cp.capacity_cny(wm, am, participation=0.10, pm=p)["capacity"].median()
    b = cp.capacity_cny(wm, am, participation=0.20, pm=p)["capacity"].median()
    assert b == pytest.approx(2.0 * a, rel=1e-9)


def test_drift_forced_trades_are_not_skipped(panel):
    """★ 目标权重恒定的买入持有组合【必须】能审，不能静默跳过。

    补测试时抓到的第三个实质缺陷：容量与不可成交都按「目标-目标之差」
    算 Δw，于是目标权重恒定时 Δw ≡ 0，每一期都被 continue 掉，
    报告只说「没有可用数据」—— 而同一份持仓的漂移调整换手是 3.09%/期。

    价格涨了就得卖掉一部分再买回来，那笔交易是真的要下单、
    真的会撞上涨跌停、真的占用成交额的。最需要审容量的情形
    （买入持有型大盘组合）恰好被跳过了。
    """
    p, reb = panel
    wm = _equal_w(p, reb)
    am = pd.DataFrame(1e8, index=p.index, columns=p.columns)
    # 目标权重逐期完全相同
    assert float((wm.iloc[1] - wm.iloc[0]).abs().max()) == 0.0
    # 但漂移调整换手明显为正
    from strategy_audit.core import turnover
    assert float(turnover(wm, p)["drift_adj"].mean()) > 0.01
    # ⇒ 两项都必须产出结果
    assert len(cp.untradable_weight(wm, p)) >= len(reb) - 2
    assert len(cp.capacity_cny(wm, am, pm=p)) >= len(reb) - 2


def test_capacity_without_prices_degrades_explicitly(panel):
    """没有价格时退回目标口径 —— 可以，但不许假装审过了。

    没有 pm 就算不出漂移，于是买入持有组合返回空 —— 这是【正确】的
    退化行为（报告会记 skip）。这里钉住的是「退化了就要空」，
    而不是悄悄用错口径给出一个看起来合理的数。
    """
    p, reb = panel
    wm = _equal_w(p, reb)
    am = pd.DataFrame(1e8, index=p.index, columns=p.columns)
    rep = AuditReport()
    cp.check_capacity(wm, am, rep)            # 不传 pm
    assert rep.skipped and not rep.findings


def test_capacity_family_never_blocks(panel):
    """★ 族五不许出 BLOCK：容量不足不会让回测净值【算错】。

    它让这份净值与你的实际规模无关 —— 一个只能管 2000 万的策略，
    回测净值本身没有错，错的是拿它去说明 20 亿的产品。
    """
    p, reb = panel
    wm = _equal_w(p, reb)
    am = pd.DataFrame(1e5, index=p.index, columns=p.columns)   # 极小成交额
    rep = AuditReport()
    cp.check_untradable(wm, p, rep)
    cp.check_capacity(wm, am, rep, pm=p)
    cp.check_liquidity_tilt(wm, am, p, rep)
    assert not rep.blockers, [f.name for f in rep.blockers]
    assert rep.findings


def test_missing_amount_skips_not_passes(panel):
    """没有成交额列时必须跳过并记录，不能报 OK。"""
    p, reb = panel
    assert cp.amount_matrix(p.stack().rename("close").reset_index()) is None
