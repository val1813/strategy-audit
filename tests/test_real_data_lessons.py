"""真实面板端到端验证抓到的两个实质缺陷，钉成回归测试。

★ 这两个都是【合成面板测不出来】的
--------------------------------
合成面板里"消失"就是永久消失、前视就是完美相关。真实 A 股面板
（800 只 / 2015-2026 / 126 期月频）两处都不长这样：

  ① 缺价 100% 是停牌（会复牌），不是退市 —— 第一版全报成退市记账 BLOCK
  ② 真实 vw 前视只有 corr=+0.088（t=4.15），毛年化却抬近 4 倍
     —— 第一版要求 |corr|≥0.10 才报，于是判了 OK

所以这里用【模拟真实结构】的合成面板复现两种情形，不依赖那份本地数据。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from strategy_audit import lookahead as la
from strategy_audit.report import BLOCK, OK, WARN, AuditReport

from synth import equal_weight, make_prices, month_ends


def _suspension_panel(n_codes=20, n_susp=4, seed=5):
    """构造【停牌后复牌】的面板：中途缺价，之后价格恢复。"""
    px = make_prices(n_codes=n_codes, seed=seed)
    n = len(px.index)
    for i in range(n_susp):
        a = int(n * (0.3 + 0.1 * i))
        px.iloc[a:a + 25, i] = np.nan          # 停牌约 25 个交易日
    return px


def test_suspension_not_reported_as_delisting_accounting():
    """★ 缺价后又复牌 ⇒ 必须说「全是停牌」，不能报退市记账政策。

    真实面板实测：24 个缺价标的-期事件 100% 复牌，第一版却给了 BLOCK。
    报错原因会让客户去修一个不存在的退市问题，然后不再相信后续告警。
    """
    px = _suspension_panel()
    wm = equal_weight(px, month_ends(px), k=8, seed=2)
    rep = AuditReport()
    la.check_membership_accounting(wm, px, rep)

    assert rep.stats["n_missing_events"] > 0, "构造未产生缺价事件，测试无效"
    assert rep.stats["n_missing_permanent"] == 0, "构造应当全是停牌"
    assert rep.stats["n_missing_resume"] > 0

    f = [f for f in rep.findings if "记账" in f.name or "停牌" in f.name][0]
    assert "停牌" in f.name, f.name
    assert f.level == WARN, "全是停牌不该 BLOCK（那是退市记账的严重度）"
    # 必须点明「全额清算」在停牌上是错的记账
    assert "复牌" in f.impact and "−100%" in f.impact


def test_permanent_disappearance_still_reported_as_accounting():
    """永久消失（真退市）仍要报记账政策区间 —— 修停牌不能把这条也关掉。"""
    px = make_prices(n_codes=20, n_dead=6, seed=3, death_drawdown=0.8)
    wm = equal_weight(px, month_ends(px), k=8, seed=2)
    rep = AuditReport()
    la.check_membership_accounting(wm, px, rep)

    assert rep.stats["n_missing_permanent"] > 0
    f = [f for f in rep.findings if "记账" in f.name][0]
    assert f.level in (BLOCK, WARN)
    assert "权威退市日" in f.impact


def test_mixed_suspension_and_delisting_counts_both():
    """停牌与退市混在一起时，两类计数都要报出来。"""
    px = make_prices(n_codes=24, n_dead=4, seed=3, death_drawdown=0.8)
    n = len(px.index)
    for i in range(6, 9):                      # 另外几只只是停牌
        a = int(n * 0.35)
        px.iloc[a:a + 25, i] = np.nan
    wm = equal_weight(px, month_ends(px), k=10, seed=2)
    rep = AuditReport()
    la.check_membership_accounting(wm, px, rep)
    assert rep.stats["n_missing_resume"] > 0
    assert rep.stats["n_missing_permanent"] > 0
    f = [f for f in rep.findings if "记账" in f.name][0]
    assert "恢复交易" in f.detail and "永久消失" in f.detail


# ---------------- ② 小而一致的前视 ----------------

def _vw_lookahead_weights(px, reb, k=30, use_next=True):
    """按【下期】收盘价（=含下期收益的"市值"代理）加权 —— 教科书 vw 前视。

    真实面板上这样做 corr 只有 +0.088，但毛年化抬近 4 倍。
    """
    rows = []
    for j, t in enumerate(reb[:-1]):
        pool = [c for c in px.columns if np.isfinite(px.loc[t, c])]
        if len(pool) < k:
            continue
        pick = pool[:k]
        ct = reb[j + 1] if use_next else t
        w = px.loc[ct, pick].astype(float)
        w = w.where(np.isfinite(w) & (w > 0))
        if w.isna().any() or float(w.sum()) <= 0:
            continue
        w = w / w.sum()
        for c, ww in w.items():
            rows.append((t, c, float(ww)))
    m = (pd.DataFrame(rows, columns=["date", "code", "weight"])
         .pivot(index="date", columns="code", values="weight")
         .fillna(0.0).sort_index())
    return m.reindex(sorted(m.columns), axis=1)


def test_small_but_consistent_lookahead_is_caught():
    """★ 判据必须是显著性，不是相关系数的绝对大小。

    30 只持仓上真前视是「小而一致」：真实面板实测 corr=+0.088 / t=4.15，
    毛年化 3.63% → 13.85%。第一版门槛 |corr|≥0.10 把它判了 OK。
    """
    px = make_prices(n_codes=60, seed=9)
    reb = month_ends(px)
    wm = _vw_lookahead_weights(px, reb, k=30, use_next=True)
    rep = AuditReport()
    la.check_weight_lookahead(wm, px, rep)

    corr = rep.stats["weight_ret_corr"]
    t = rep.stats["weight_ret_corr_t"]
    assert t > 2.0, f"构造未产生显著前视（t={t:.2f}），测试无效"
    assert rep.blockers or rep.warnings, (
        f"corr={corr:+.3f} t={t:.2f} 的前视被判 OK —— 幅度门槛又把它放过了")
    # 报告必须说明「幅度小不等于影响小」
    f = (rep.blockers or rep.warnings)[0]
    assert "幅度小不等于影响小" in f.impact


def test_negative_consistent_correlation_is_not_called_lookahead():
    """显著为【负】不是前视（前视偏向当期涨的票）—— 报 OK 但说明白。"""
    px = make_prices(n_codes=60, seed=9)
    reb = month_ends(px)
    # 按下期价格【倒序】加权 ⇒ 系统性偏向跌的票
    rows = []
    for j, t in enumerate(reb[:-1]):
        pool = [c for c in px.columns if np.isfinite(px.loc[t, c])][:30]
        if len(pool) < 30:
            continue
        v = px.loc[reb[j + 1], pool].astype(float)
        w = (1.0 / v)
        w = w / w.sum()
        for c, ww in w.items():
            rows.append((t, c, float(ww)))
    m = (pd.DataFrame(rows, columns=["date", "code", "weight"])
         .pivot(index="date", columns="code", values="weight")
         .fillna(0.0).sort_index())
    wm = m.reindex(sorted(m.columns), axis=1)
    rep = AuditReport()
    la.check_weight_lookahead(wm, px, rep)
    if rep.stats.get("weight_ret_corr_t", 0) < -2:
        f = [f for f in rep.findings if "权重前视" in f.name][0]
        assert f.level == OK
        assert "与前视方向相反" in f.detail
