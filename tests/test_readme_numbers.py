"""README 里的每个数字都必须能复现。

★ 为什么单开一个文件
------------------
factor-audit 的实测教训：发版前发现三处 README 陈述已经过时，
而代码和测试都是绿的 —— 因为没有任何东西把文档里的数字钉住。
文档数字漂移在客户那里就是「工具在撒谎」。

所以 README 引用的每个测量值在这里断言一次。改了实现导致数字变了，
这个文件会红 —— 提醒你【同时改 README】，而不是偷偷让文档过期。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from strategy_audit import audit_strategy, core, lookahead as la
from strategy_audit.report import AuditReport

from synth import equal_weight, make_prices, month_ends, to_long

TOL = 5e-4


def test_readme_static_weight_turnover():
    """§1「什么都没改」不等于「不用交易」：朴素 0.00% vs 漂移 2.59%。"""
    px = make_prices(n_codes=30, n_dead=6, seed=3)
    reb = month_ends(px)
    codes = list(px.columns[:10])
    wm = pd.DataFrame(0.1, index=pd.Index(reb, name="date"), columns=codes)
    to = core.turnover(wm, px[codes])
    assert float(to["naive"].abs().max()) < 1e-15
    assert abs(float(to["drift_adj"].mean()) - 0.0259) < TOL


def test_readme_implied_turnover_ratio():
    """§2 反推 89.0% / 实测 78.6% / 比值 1.13。"""
    px = make_prices()
    wm = equal_weight(px, month_ends(px))
    implied = core.rank_autocorr_turnover(wm)
    actual = float(core.turnover(wm, px)["drift_adj"].mean())
    assert abs(implied - 0.890) < TOL, implied
    assert abs(actual - 0.786) < TOL, actual
    assert abs(implied / actual - 1.13) < 0.005


def test_readme_policy_spread():
    """§3 三政策 +30.68% / +22.36% / +0.05%，跨度 30.63pp（退市前跌 80%）。"""
    px = make_prices(n_codes=30, n_dead=6, seed=3, death_drawdown=0.8)
    reb = month_ends(px)
    wm = equal_weight(px, reb, k=8, seed=2)
    tot = {p: float((1.0 + core.period_returns(wm, px, policy=p)["ret"]).prod()) - 1.0
           for p in core.MISSING_POLICIES}
    assert abs(tot["drop"] - 0.3068) < TOL, tot
    assert abs(tot["hold_last"] - 0.2236) < TOL, tot
    assert abs(tot["zero"] - 0.0005) < TOL, tot
    spread = (max(tot.values()) - min(tot.values())) * 100
    assert abs(spread - 30.63) < 0.05, spread

    pr = core.period_returns(wm, px)
    assert int((pr["n_missing"] > 0).sum()) == 2
    assert abs(float(pr["w_missing"].max()) - 0.125) < 1e-9


def test_readme_policy_order_flips_without_crash():
    """§3 的前提：无崩盘时 6 个面板里 3 个出现 drop < hold_last。"""
    flips = 0
    for nc, nd, k, sd in [(20, 6, 8, 3), (30, 6, 8, 3), (40, 8, 10, 3),
                          (25, 5, 6, 7), (30, 10, 8, 3), (50, 12, 10, 5)]:
        px = make_prices(n_codes=nc, n_dead=nd, seed=sd, death_drawdown=0.0)
        wm = equal_weight(px, month_ends(px), k=k, seed=2)
        tot = {p: float((1.0 + core.period_returns(wm, px, policy=p)["ret"]).prod())
               for p in core.MISSING_POLICIES}
        if tot["drop"] < tot["hold_last"]:
            flips += 1
    assert flips == 4, f"README 写 4 个翻转，实测 {flips}"


def test_readme_day0_mde_on_clean_portfolio():
    """§4 干净组合：每期 +0.116%（相对 +10.4%）、t=0.75、MDE 0.301%/期，判 OK。"""
    px = make_prices(n_codes=60, start="2021-01-04", end="2023-12-29",
                     seed=5, sigma=0.02)
    wm = equal_weight(px, month_ends(px), k=10, seed=7)
    rep = AuditReport()
    la.check_rebalance_alignment(wm, px, rep)

    gain = (rep.stats["ret_mean_eat_day0"] - rep.stats["ret_mean_asis"]) * 100
    rel = (rep.stats["ret_mean_eat_day0"] / rep.stats["ret_mean_asis"] - 1) * 100
    assert abs(gain - 0.116) < 5e-3, gain
    assert abs(rel - 10.4) < 0.15, rel
    assert abs(rep.stats["day0_t"] - 0.75) < 0.01
    assert abs(rep.stats["day0_mde"] * 100 - 0.301) < 5e-3
    # README 的论点就是「这不该报警」
    assert not rep.blockers and not rep.warnings


def test_readme_reconciliation_precision():
    """核心数学一节：对账偏差 < 1e-12（README 写实测 4.5e-16）。"""
    px = make_prices()
    wm = equal_weight(px, month_ends(px))
    pr = core.period_returns(wm, px, policy="hold_last")
    dp = core.daily_path(wm, px)
    worst = max(abs(float((1.0 + dp[(dp.index > t0) & (dp.index <= r["end"])])
                          .prod() - 1.0) - float(r["ret"]))
                for t0, r in pr.iterrows())
    assert worst < 1e-12
    assert worst < 1e-14, f"README 声称 4.5e-16 量级，实测 {worst:.2e}"


def test_readme_planted_fee_recovered_exactly():
    """核心数学一节：植入 15bp，对账必须反推出 15bp。"""
    px = make_prices()
    wm = equal_weight(px, month_ends(px))
    from strategy_audit import turnover_cost as tc
    to = core.turnover(wm, px)
    pr = core.period_returns(wm, px)
    t = to["drift_adj"].reindex(pr.index).fillna(0.0)
    net = pr["ret"] - 2.0 * 15e-4 * t
    rep = AuditReport()
    tc.check_gross_net_reconcile(pr["ret"], net, to, rep)
    assert abs(rep.stats["implied_cost_bp"] - 15.0) < 1e-6


def test_readme_test_count_matches():
    """README 写「67 个测试」—— 数量变了就得改文档。

    ★ factor-audit 的发版清单第一条就是这个：改了测试数必须同步 README。
    """
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    out = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q"],
                         cwd=root, capture_output=True, text=True).stdout
    total = sum(int(ln.split(":")[1].strip())
                for ln in out.splitlines()
                if ln.startswith("tests/") and ":" in ln)
    en = (root / "README.md").read_text(encoding="utf-8")
    zh = (root / "README.zh-CN.md").read_text(encoding="utf-8")
    assert f"{total} tests" in en, (
        f"实际 {total} 个测试，英文 README 未同步")
    assert f"{total} 个测试" in zh, (
        f"实际 {total} 个测试，中文 README 未同步")


# ---------------- 「你有什么→能审什么」那张表 ----------------

def test_readme_capability_counts():
    """README 那张表的三个计数必须与 CHECKS 一致（加检查就得改 README）。"""
    from strategy_audit import capability as cap
    readme = _readme_zh()
    n_nav = len(cap.available({cap.NAV})[0])
    n_wp = len(cap.available({cap.W, cap.P, cap.NAV})[0])
    n_all = len(cap.CHECKS)
    # 计数从 CHECKS 推导；README 必须跟着 CHECKS 走，不是反过来
    assert f"**{n_nav} 项**" in readme, f"README 未写 {n_nav} 项(只有净值)"
    assert f"**{n_wp} 项**" in readme, f"README 未写 {n_wp} 项(权重+价格)"
    assert f"**{n_all} 项**" in readme, f"README 未写 {n_all} 项(全部)"
    assert f"{n_all - n_wp} 项（缺可选输入）" in readme


def test_readme_demo_capability_claim():
    """--demo 报的「能审 N/M 项」必须是真的，缺的那几项都只缺可选输入。"""
    from strategy_audit.cli import _demo_inputs
    from strategy_audit import audit, capability as cap
    w, p = _demo_inputs()
    rep = audit(w, p, show_detection=False)
    have = set(rep.stats["capability"])
    ok, no = cap.available(have)
    assert len(ok) == len(cap.CHECKS) - len(no)
    # ★ 从 CHECKS 推导，不写死具体是哪个可选输入（族五加了成交额列时
    # 写死 cap.NET 的版本就红了，而它跟族五无关）
    optional = {cap.NET, cap.BENCH, cap.AMT, cap.OWN_NAV, cap.OPEN, cap.FLAGS,
                cap.SIG, cap.SIG_ALT, cap.DELISTED, cap.PARAMS, cap.TRADES}
    for c in no:
        lack = set(c.needs) - have
        assert lack and lack <= optional, (c.key, lack)


# ---------------- 六个「工具自己被抓到的错」 ----------------

def test_readme_conc_false_positive_claim():
    """§③ 固定门槛 0.50 误报 61%、零分布中位数 0.53。"""
    import numpy as np
    idx = pd.date_range("2016-01-31", periods=120, freq="ME")
    tops = []
    for s in range(400):
        r = pd.Series(np.random.default_rng(s).normal(0.006, 0.045, 120),
                      index=idx)
        by = r.groupby(r.index.year).apply(lambda x: float((1 + x).prod() - 1))
        pos = by[by > 0]
        if len(pos) >= 2 and float((1 + r).prod() - 1) > 0:
            tops.append(float(pos.nlargest(2).sum() / pos.sum()))
    t = np.array(tops)
    assert abs(float(np.median(t)) - 0.53) < 0.02, float(np.median(t))
    assert abs(float((t > 0.5).mean()) - 0.61) < 0.04, float((t > 0.5).mean())


def test_readme_single_period_null_claim():
    """§④ 「单期占累计」纯噪声 p90=0.78 / p95=1.45。"""
    import numpy as np
    idx = pd.date_range("2016-01-31", periods=120, freq="ME")
    sh = []
    for s in range(400):
        r = pd.Series(np.random.default_rng(s).normal(0.006, 0.045, 120),
                      index=idx)
        tot = float((1 + r).cumprod().iloc[-1] - 1)
        if abs(tot) > 1e-9:
            sh.append(abs(float(r.max()) / tot))
    a = np.array(sh)
    assert abs(float(np.quantile(a, .9)) - 0.78) < 0.06
    assert abs(float(np.quantile(a, .95)) - 1.45) < 0.20


def test_readme_nw_crossing_claim():
    """§⑤ 16% 的纯噪声跨过 |t|=2，跨度中位数 0.57。"""
    import numpy as np
    from strategy_audit import significance as sg
    idx = pd.date_range("2016-01-31", periods=120, freq="ME")
    cross, gaps = 0, []
    N = 200
    for s in range(N):
        r = pd.Series(np.random.default_rng(5000 + s).normal(0.006, 0.045, 120),
                      index=idx)
        ts = {L: sg.newey_west_t(r.values, L) for L in range(13)}
        ts = {k: v for k, v in ts.items() if np.isfinite(v)}
        lo, hi = min(abs(v) for v in ts.values()), max(abs(v) for v in ts.values())
        if lo < 2.0 <= hi:
            cross += 1
            gaps.append(hi - lo)
    assert abs(cross / N - 0.16) < 0.03, cross / N
    assert abs(float(np.median(gaps)) - 0.57) < 0.08


def test_readme_sharpe_se_agreement_claim():
    """§⑥ 修好后两个独立估计差异 < 1.2%（4 组种子）。"""
    from strategy_audit import significance as sg
    from strategy_audit.core import annualize
    import numpy as np
    idx = pd.date_range("2016-01-31", periods=120, freq="ME")
    worst = 0.0
    for seed in (3, 11, 7, 21):
        r = pd.Series(np.random.default_rng(seed).normal(0.006, 0.045, 120),
                      index=idx)
        sr = annualize(r, 12.0)["sharpe"]
        se = sg.deflated_sharpe(sr, len(r), 1, 12.0)["se"]
        t_nw = sg.newey_west_t(r.values, 0)
        worst = max(worst, abs(sr / se - t_nw) / abs(t_nw))
    assert worst <= 0.013, f"实测最大差异 {worst:.2%}，README 写 ≤1.3%"


# ---------------- 族六：处方层的三条 README 主张 ----------------

def test_readme_identity_drift_claim():
    """族六「身份必须钉住」那张表：不分类削预算会把组合换掉。

    README 用真实面板的数字（φ=0.4 时最大权重 95.7%、注数 1.5、
    名单重合 1%）。这里在合成面板上复现【同一个机制】：
    按 |Δw| 排序削预算 ⇒ 名单重合崩塌；分类后 ⇒ 恒为 100%。
    """
    import numpy as np
    from strategy_audit import prescribe as pr
    from synth import make_prices, month_ends, tilted_weight

    p = make_prices(n_codes=60, seed=5)
    reb = month_ends(p)
    wm = tilted_weight(p, reb, k=12)

    # 分类版（当前实现）：名单重合恒 100%
    for phi in (0.6, 0.4, 0.2):
        assert pr.identity_controlled_path(wm, p, phi)["overlap"] == \
            pytest.approx(1.0, abs=1e-9), phi

    # 不分类版（第一版的做法）：名单重合会崩
    from strategy_audit.core import _seg_gross, align, drift_weights
    wmA, pmA = align(wm, p)
    dates = list(wmA.index)
    hold = wmA.loc[dates[0]].copy()
    ovs = []
    for cur, nxt in zip(dates[:-1], dates[1:]):
        g = _seg_gross(pmA, cur, nxt)
        drift = drift_weights(hold, g)
        tgt = wmA.loc[nxt]
        dw = tgt - drift
        limit, used = 0.4 * float(dw.abs().sum()), 0.0
        new = drift.copy()
        for c, v in dw.abs().sort_values(ascending=False).items():
            if used + v > limit:
                break
            new[c] = tgt[c]
            used += v
        gr = float(new.abs().sum())
        new = new / gr if gr > 0 else new
        nz = new[new.abs() > 1e-12]
        tn = tgt[tgt.abs() > 1e-12].index
        ovs.append(len(set(nz.index) & set(tn)) / max(len(tn), 1))
        hold = new
    assert float(np.mean(ovs)) < 0.90, (
        f"不分类削预算的名单重合 {np.mean(ovs):.0%} —— "
        "README 声称它会崩塌，若没崩就是这条主张站不住了")


def test_readme_tweak_share_gap_claim():
    """族六「7% 与 88% 之间没有任何形态」：门槛落在空档里。"""
    import numpy as np
    from strategy_audit import prescribe as pr
    from synth import equal_weight, make_prices, month_ends
    readme = _readme_zh()
    assert f"微调占换手 ≥ {pr.TWEAK_SHARE_FLOOR:.0%}" in readme

    p = make_prices(n_codes=60, seed=5)
    reb = month_ends(p)
    churny = pr.split_turnover(equal_weight(p, reb, k=12), p)["share"]
    codes = sorted(p.columns[p.loc[reb[-1]].notna()])[:12]
    w = np.linspace(0.16, 0.02, len(codes))
    fixed = pd.DataFrame([w / w.sum()] * len(reb),
                         index=pd.DatetimeIndex(reb), columns=codes)
    tweaky = pr.split_turnover(
        fixed.reindex(columns=sorted(p.columns), fill_value=0.0), p)["share"]
    # 两类形态必须被门槛分开，且各自离门槛都有余量
    assert churny < pr.TWEAK_SHARE_FLOOR - 0.10 < \
        pr.TWEAK_SHARE_FLOOR + 0.10 < tweaky, (churny, tweaky)


def test_readme_documents_every_family():
    """★ 每一族都必须在 README 里有小节 —— 加了族却不写文档等于藏起来。

    ★ 族数从 CHECKS 推导，不写死「六族」：实测加族七时这个测试
    因为硬编码了「## 六族检查」而失败，而它本该只关心「都写了没」。
    """
    from strategy_audit import capability as cap
    readme = _readme_zh()
    secs = {c.section for c in cap.CHECKS}
    # 标题里的族数必须与实际族数一致（输入契约不算一族）
    n_fam = len(secs - {"输入契约"})
    zh = "一二三四五六七八九十"[n_fam - 1] if n_fam <= 10 else str(n_fam)
    assert f"## {zh}族检查" in readme, f"README 标题没写 {n_fam} 族"
    for sec in secs:
        assert sec in readme, f"README 没有介绍「{sec}」这一族"


def _readme():
    from pathlib import Path
    return (Path(__file__).resolve().parent.parent / "README.md").read_text(
        encoding="utf-8")


def _readme_zh():
    from pathlib import Path
    return (Path(__file__).resolve().parent.parent / "README.zh-CN.md").read_text(
        encoding="utf-8")


def test_readme_language_links():
    """中英文 README 必须互相链接（英文主版 + 中文版本）。"""
    en = _readme()
    zh = _readme_zh()
    assert "README.zh-CN.md" in en, "英文 README 没有链接到中文版本"
    assert "README.md" in zh, "中文 README 没有链接到英文版本"
