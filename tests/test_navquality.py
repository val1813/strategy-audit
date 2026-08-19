"""族七：净值质量。只要一条曲线就能跑 —— 这一族是为了【扩大适用面】。

前六族里门槛最低的族三也只审「显著性」，而它默认那条曲线是真的。
机构拿到的第一份东西往往【只有净值】，那时能审的 4 项全都建立在
「曲线本身可信」之上。族七补的就是那个前提。

★ 本族的判据必须同时满足两条，缺一不可
------------------------------------
    统计显著   t = AC1·√n ≥ 2      （否则是噪声）
    经济重要   解平滑放大 ≥ 1.15x   （否则没有决策含义）
只看 t：长样本下 AC1=+0.048 也能 t=2.47 ⇒ 每条日频净值都被报可疑。
只看放大：短样本噪声能凑出 1.2x ⇒ 误报。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from strategy_audit import navquality as nq
from strategy_audit.report import AuditReport


def _daily(n=1200, seed=3, mu=0.0004, sd=0.014):
    idx = pd.bdate_range("2016-01-04", periods=n)
    return pd.Series(np.random.default_rng(seed).normal(mu, sd, n), index=idx)


# ---------------- 解平滑的数学性质 ----------------

def test_unsmooth_returns_one_when_no_autocorr():
    """AC1 ≤ 0 时不许「放大」波动 —— 那会把 Sharpe 无端调低。"""
    for seed in (1, 5, 9):
        amp, th = nq.unsmooth_amplification(_daily(seed=seed).values)
        assert amp == pytest.approx(1.0, abs=0.25)
        assert 0.0 <= th < 0.5


def test_unsmooth_recovers_planted_moving_average():
    """★ 对已知的 MA 平滑，解平滑要能还原出放大倍数。

    造 r_obs = 0.5·r(t) + 0.5·r(t−1)（θ1=0.5，理论放大 1/√0.5 = 1.414），
    解平滑应当给出接近 1.41 的倍数 —— 这是可对账的解析值。
    """
    r = _daily(n=4000, seed=11)
    obs = (0.5 * r + 0.5 * r.shift(1)).dropna()
    amp, th1 = nq.unsmooth_amplification(obs.values)
    assert amp == pytest.approx(1.0 / np.sqrt(0.5), rel=0.05), amp
    assert th1 == pytest.approx(0.5, abs=0.05), th1


def test_unsmooth_is_a_lower_bound_at_higher_order():
    """★ 一阶解平滑对高阶平滑给出的是【下界】，不能反过来高估。

    3 期等权 MA 的真实放大是 1/√(3·(1/3)²) = √3 ≈ 1.73。
    一阶闭式解会给出比它小的数 —— 那是刻意的（见模块 docstring：
    高阶拟合会引入模型选择自由度）。这里钉住方向：只能偏小。
    """
    r = _daily(n=4000, seed=13)
    obs = r.rolling(3).mean().dropna()
    amp, _ = nq.unsmooth_amplification(obs.values)
    assert 1.15 < amp < np.sqrt(3.0) + 1e-9, amp


def test_ac1_null_sd_scales_as_one_over_sqrt_n():
    """★ 零分布标准差 ≈ 1/√n —— 这是「门槛必须随样本量缩放」的依据。

    实测 2000 次纯噪声：n=60 sd 0.124（1/√n=0.129）、
    n=120 sd 0.091（0.091）、n=250 sd 0.063（0.063）。
    所以固定的相关系数门槛必然在一端出错，判据只能是 t = AC1·√n。
    """
    for n in (60, 250):
        rng = np.random.default_rng(11)
        a = np.array([nq.ac1(rng.normal(0.006, 0.045, n)) for _ in range(400)])
        assert a.std() == pytest.approx(1.0 / np.sqrt(n), rel=0.25), n
        assert abs(a.mean()) < 3.0 / np.sqrt(n)


# ---------------- ① 平滑：两个条件都要满足 ----------------

def test_long_clean_daily_series_is_not_flagged():
    """★ 长样本 + 极小自相关 ⇒ 必须放行。

    盲测那份真实日频净值 n=2645、AC1=+0.048 ⇒ t=2.47 统计显著，
    但解平滑放大仅 1.05x，Sharpe 0.29→0.28 没有决策含义。
    只看 t 会把每一条日频净值都报成可疑 —— 那样的报警没人看。
    """
    r = _daily(n=2600, seed=21)
    # 掺入极弱的自相关，使 t 显著但放大微弱
    obs = (r + 0.05 * r.shift(1)).dropna()
    rep = AuditReport()
    out = nq.check_smoothing(obs, 252.0, rep)
    assert out["amp"] < nq.UNSMOOTH_WARN
    assert not rep.warnings, rep.findings[0].detail


@pytest.mark.parametrize("mu,seed", [(0.0009, 23), (-0.0004, 23)])
def test_planted_smoothing_is_flagged_with_sharpe_impact(mu, seed):
    """★ 真的平滑过必须报出来，并给出 |Sharpe| 的修正方向。

    解平滑放大的是【波动】，所以 |Sharpe| 必然变小 —— 但符号不能变，
    也不能把负 Sharpe 说成「被高估」。实测抓到的 bug：
    Sharpe −0.24 除以 1.41 得 −0.17，报告写成「Sharpe 应按 −0.17 计」，
    等于把一个亏钱的策略修饰成亏得少一点。
    """
    r = _daily(n=1200, seed=seed, mu=mu)
    obs = r.rolling(3).mean().dropna()
    rep = AuditReport()
    out = nq.check_smoothing(obs, 252.0, rep)
    assert out["t"] > nq.AC1_T_WARN and out["amp"] > nq.UNSMOOTH_WARN
    assert rep.warnings

    sr = rep.stats["nav_sharpe_reported"]
    adj = rep.stats["nav_sharpe_unsmoothed"]
    assert abs(adj) < abs(sr)                    # |Sharpe| 必须变小
    assert np.sign(adj) == np.sign(sr)           # 符号不许翻
    txt = rep.warnings[0].impact
    if sr > 0:
        assert "Sharpe 被高估" in txt
    else:
        # 负 Sharpe 时要说「波动被低估」，不许声称收益被高估
        assert "波动被低估" in txt
        assert "Sharpe 被高估" not in txt


def test_short_series_skips_not_passes():
    """点数不足时跳过并记录 —— 跳过 ≠ 查过通过。"""
    rep = AuditReport()
    nq.check_smoothing(_daily(n=10), 252.0, rep)
    assert rep.skipped and not rep.findings


# ---------------- ② 停滞估值 ----------------

def test_stale_detects_weekly_valuation_in_daily_clothing():
    """★ 每 5 日才更新估值 ⇒ 停滞占比应约 80%。

    这类曲线的日频 Sharpe 是没有意义的 —— 报告必须指出要按
    真实估值频率重算。
    """
    r = _daily(n=1000, seed=31)
    v = r.values.copy()
    for i in range(len(v)):
        v[i] = r.values[max(0, i - 4):i + 1].sum() if i % 5 == 0 else 0.0
    nav = (1 + pd.Series(v, index=r.index)).cumprod()
    rep = AuditReport()
    out = nq.check_stale(nav, "nav", rep)
    assert out["stale"] > 0.7
    assert rep.warnings
    assert "真实估值频率" in rep.warnings[0].impact


def test_stale_zero_on_continuous_series():
    """连续分布下零收益概率为 0 —— 干净序列必须报 OK。"""
    rep = AuditReport()
    out = nq.check_stale(_daily(n=600, seed=33), "ret", rep)
    assert out["stale"] == 0.0
    assert not rep.warnings


def test_stale_reads_nav_not_recomputed_returns():
    """★ 停滞要看【原始自报序列】。

    重算出来的收益曲线不会有「相邻点相等」这个特征 —— 拿它去查停滞
    永远查不出东西。这一条钉住 role 的语义：nav 看差分、ret 看本身。
    """
    idx = pd.bdate_range("2016-01-04", periods=100)
    flat = pd.Series(np.repeat(np.arange(1.0, 21.0), 5), index=idx)
    assert nq.stale_share(flat, "nav") == pytest.approx(0.8, abs=0.02)
    # 同一份数据当成收益率读，语义完全不同（不该也是 0.8）
    assert nq.stale_share(flat, "ret") != pytest.approx(0.8, abs=0.02)


# ---------------- ③ 期末修饰 ----------------

def test_dressing_null_t_is_well_behaved():
    """★ 门槛 2.5 的依据：t 的零分布 sd ≈ 1.0、|t|>2.5 的比例仅 1~2%。

    实测（400 次纯噪声 × 三个样本长度）：
        n= 750  sd 1.03  |t|>2 7.0%  |t|>2.5 2.0%
        n=1500  sd 1.00  |t|>2 3.8%  |t|>2.5 1.8%
        n=2600  sd 1.04  |t|>2 5.0%  |t|>2.5 0.8%
    2.0 的误报率偏高（最坏 7%），2.5 才把它压到 2% 以下。
    """
    ts = []
    for s in range(120):
        rep = AuditReport()
        o = nq.check_period_end_dressing(_daily(n=1500, seed=2000 + s), rep)
        if o and np.isfinite(o.get("t", np.nan)):
            ts.append(o["t"])
    ts = np.array(ts)
    assert ts.std() == pytest.approx(1.0, abs=0.25)
    assert (np.abs(ts) > nq.DRESS_T_WARN).mean() <= 0.06


def test_dressing_detects_planted_month_end_lift():
    """★ 月末最后一日人为抬高必须被抓到。

    ★ 植入幅度必须显著超过本样本的最小可检出效应。
    第一版植 +40bp，而 n=1500、70 个月末点、日 sd=1.42% 的样本
    MDE 就是 29bp —— t 只有 2.12，判 OK 是【正确的】。
    测试植入一个查不出的效应然后要求查出来，那是测试写错了，不是代码错了。
    """
    r = _daily(n=1500, seed=41)
    me = r.groupby([r.index.year, r.index.month]).tail(1).index
    r.loc[me] += 0.012                      # 120bp，约 4 倍于 MDE
    rep = AuditReport()
    out = nq.check_period_end_dressing(r, rep)
    assert out["t"] > nq.DRESS_T_WARN, out
    assert rep.warnings and "月末" in rep.warnings[0].name


def test_dressing_clean_series_reports_mde():
    """★ 没查到时必须给最小可检出效应，而不是只说「正常」。"""
    rep = AuditReport()
    nq.check_period_end_dressing(_daily(n=1500, seed=43), rep)
    assert not rep.warnings
    assert "最小可检出" in rep.findings[0].detail


def test_dressing_skips_on_monthly_series():
    """★ 月频序列每个点本身就是月末 ⇒ 没有期末/期内之分，必须跳过。

    硬算会拿「月末点」和空集比，或者把全样本当成期末 —— 两种都是
    在造一个没有意义的 t 值。
    """
    idx = pd.date_range("2016-01-31", periods=120, freq="ME")
    r = pd.Series(np.random.default_rng(7).normal(.006, .045, 120), index=idx)
    rep = AuditReport()
    assert nq.check_period_end_dressing(r, rep) == {}
    assert rep.skipped and not rep.findings


@pytest.mark.parametrize("name,series", [
    ("含inf", [1.0] * 30 + [np.inf] + [1.0] * 29),
    ("含-inf", [1.0] * 30 + [-np.inf] + [1.0] * 29),
    ("全常数", [1.0] * 60),
    ("含nan", [np.nan] * 30 + [1.0] * 30),
    ("极端值", [1e300] * 30 + [1e-300] * 30),
    ("负净值", list(np.linspace(1, -1, 60))),
])
def test_no_nan_leaks_into_report(name, series):
    """★ 敌意输入下报告里绝不许出现 nan。

    这条抓到过一个实质缺陷：序列含一个 ±inf 时 AC1 仍算得出有限值，
    但 Sharpe 变成 nan，于是报告打出「解平滑后波动放大 1.00x ⇒
    Sharpe nan → nan」还给了 OK 结论 —— 一份看起来像模像样、
    实际什么都没审的报告。

    根因是 `dropna()` 拦不住 ±inf（pandas 的老坑）。本族每个入口
    现在都用 isfinite 过滤。上游 _clean_series 也会清，但公开 API
    必须自己站得住。
    """
    idx = pd.date_range("2016-01-31", periods=60, freq="ME")
    s = pd.Series(series, index=idx, dtype=float)
    for role in ("nav", "ret"):
        rep = AuditReport()
        r = s.pct_change().dropna() if role == "nav" else s.dropna()
        # 极端数值是测试输入，不是允许冒出 RuntimeWarning 的理由。
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            nq.check_smoothing(r, 12.0, rep)
            nq.check_stale(s, role, rep)
            nq.check_period_end_dressing(r, rep)
        txt = rep.text().lower().replace("nan/", "")
        assert "nan" not in txt, (name, role)
        assert not rep.blockers
        # 必须给出结论或明确跳过，不能两手空空
        assert rep.findings or rep.skipped, (name, role)


def test_dressing_needs_datetime_index():
    """索引不是日期时跳过，不许崩。"""
    rep = AuditReport()
    assert nq.check_period_end_dressing(
        pd.Series(np.zeros(100)), rep) == {}
    assert rep.skipped


# ---------------- 全族：判别力与不出 BLOCK ----------------

@pytest.mark.parametrize("kind,expect", [
    ("clean_daily", 0),          # 干净日频：全 OK
    ("clean_monthly", 0),        # 干净月频：全 OK（期末项跳过）
    ("smoothed", 1),
    ("stale", 1),
    ("dressed", 1),
])
def test_family_seven_discriminates(kind, expect):
    """★ 端到端判别力：两个对照静默，三种植入缺陷各报一次。"""
    from strategy_audit import audit
    rng = np.random.default_rng(7)
    r = _daily(n=1500, seed=51)
    if kind == "clean_daily":
        s = (1 + r).cumprod()
    elif kind == "clean_monthly":
        idx = pd.date_range("2016-01-31", periods=120, freq="ME")
        s = (1 + pd.Series(rng.normal(.006, .045, 120), index=idx)).cumprod()
    elif kind == "smoothed":
        s = (1 + r.rolling(3).mean().dropna()).cumprod()
    elif kind == "stale":
        v = r.values.copy()
        for i in range(len(v)):
            v[i] = r.values[max(0, i - 4):i + 1].sum() if i % 5 == 0 else 0.0
        s = (1 + pd.Series(v, index=r.index)).cumprod()
    else:
        me = r.groupby([r.index.year, r.index.month]).tail(1).index
        r.loc[me] += 0.012          # 见 test_dressing_*：须显著超过 MDE
        s = (1 + r).cumprod()
    rep = audit(s, show_detection=False)
    warns = [f.name for f in rep.findings
             if f.section == "净值质量" and f.level == "WARN"]
    assert len(warns) == expect, (kind, warns)


def test_family_seven_never_blocks():
    """★ 族七不出 BLOCK。

    自相关高有正当成因（含估值滞后的资产、数据重叠），做成 BLOCK
    会逼用户为了过闸门去改一条可能本来就没错的净值。
    """
    from strategy_audit import audit
    r = _daily(n=1500, seed=61)
    for s in ((1 + r.rolling(5).mean().dropna()).cumprod(),
              (1 + r).cumprod()):
        rep = audit(s, show_detection=False)
        f7 = [f for f in rep.findings if f.section == "净值质量"]
        assert f7
        assert not [f for f in f7 if f.level == "BLOCK"]
