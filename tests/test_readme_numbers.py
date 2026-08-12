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
    readme = (root / "README.md").read_text(encoding="utf-8")
    assert f"{total} 个测试" in readme, (
        f"实际 {total} 个测试，README 未同步")
