"""顶层 API：把两族检查串成一份报告。

    from strategy_audit import audit_strategy
    rep = audit_strategy(weights, prices)
    print(rep.text())
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import lookahead as la
from . import turnover_cost as tc
from .contract import check_gross, load_prices, load_weights, normalize_gross, to_matrix
from .core import period_returns, periods_per_year, price_matrix
from .report import BLOCK, AuditReport


def audit_strategy(weights: pd.DataFrame,
                   prices: pd.DataFrame,
                   *,
                   net_returns: pd.Series | None = None,
                   benchmark: pd.Series | None = None,
                   name: str = "策略审计") -> AuditReport:
    """审一份回测：权重面板 + 价格面板。

    weights      date, code, weight —— 每个调仓日的目标权重
    prices       date, code, close  —— 日频价格（与 factor-audit 同一契约）
    net_returns  可选。你自己算的【净】收益（index=调仓日），用于毛净对账
    benchmark    可选。基准收益（index=调仓日），盈亏平衡成本按超额算

    ★ 检查顺序：契约 → 前视与记账 → 换手与成本。
    前视排在成本前面，因为前视是「这份净值是不是真的」，
    成本是「这份净值扣够了没有」—— 前者不成立时后者没有意义。
    这与 factor-audit「面板体检必须在审因子之前」是同一条道理。
    """
    rep = AuditReport(title=name)

    w = load_weights(weights, rep)
    p = load_prices(prices, rep)
    if w is None or p is None:
        return rep

    w = check_gross(w, rep)
    wm = normalize_gross(to_matrix(w))
    pm = price_matrix(p)

    rep.stats.update(
        n_dates=len(wm.index),
        n_names=int((wm != 0).any().sum()),
        span=f"{wm.index.min().date()} ~ {wm.index.max().date()}",
    )

    if len(wm.index) < 3:
        rep.add(BLOCK, "调仓期数不足", f"仅 {len(wm.index)} 个调仓日",
                "至少需要 3 期才能算换手；多数检查需要 6~7 期",
                section="输入契约")
        return rep

    # ---- 族二：前视与记账（先跑）----
    la.check_rebalance_alignment(wm, pm, rep)
    la.check_weight_lookahead(wm, pm, rep)
    la.check_universe_survivorship(wm, pm, rep)
    la.check_membership_accounting(wm, pm, rep)

    # ---- 族一：换手与成本 ----
    to = tc.check_turnover_basis(wm, pm, rep)
    tc.check_implied_turnover(wm, to, rep)

    pr = period_returns(wm, pm)
    ppy = periods_per_year(wm.index)
    rep.stats["periods_per_year"] = ppy
    tc.check_breakeven(pr["ret"], to, ppy, rep, bench=benchmark)

    if net_returns is not None:
        tc.check_gross_net_reconcile(pr["ret"], net_returns, to, rep)
    else:
        rep.skip("毛净对账", "未提供 net_returns，无法反推你实际计了多少 bp")

    return rep
