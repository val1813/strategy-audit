"""strategy_audit —— 策略审计器。

审查【回测到净值这一段有没有偷分】，而不是告诉你策略好不好。

    from strategy_audit import audit_strategy

    rep = audit_strategy(weights, prices)      # 权重面板 + 价格面板
    print(rep.text())

★ 与 factor-audit 的分工
    factor-audit    审信号层：检验做没做对（IC、NW t、独立簇 Bonferroni、衰减）
    strategy-audit  审组合层：从信号到净值这一段（换手成本、前视、成分记账）
  两者共用 date/code/close 价格面板契约，可以串起来用。

★ 两族缺陷
    换手与成本   朴素 vs 漂移调整口径、反推换手偏差、盈亏平衡成本、毛净对账
    前视与记账   调仓日对齐、权重前视、股票池生存者、成分变动三政策区间

★ 三件它不做的事
    · 不告诉你策略好不好 —— 只告诉你这份净值能支撑什么结论
    · 不替你选成本假设 —— 主输出是盈亏平衡成本，那不需要假设
    · 不上传任何东西 —— 全本地运行，持仓不出你的进程

★ 数据由你提供。与 factor-audit 同一条原则：不内置数据源
  （实测免费源会拉黑 IP、非线程安全、对退市股返回「空 + success」），
  也不分发行情（版权）。
"""

from __future__ import annotations

from ._api import audit, audit_strategy
from .capability import CHECKS, available, matrix_text, missing_value
from .contract import check_gross, load_prices, load_weights, normalize_gross, to_matrix
from .detect import Detected, classify_series, detect_all, detect_frame
from .core import (
    MISSING_POLICIES,
    align,
    annualize,
    daily_path,
    drift_weights,
    period_returns,
    periods_per_year,
    price_matrix,
    rank_autocorr_turnover,
    turnover,
)
from .lookahead import (
    check_membership_accounting,
    check_rebalance_alignment,
    check_universe_survivorship,
    check_weight_lookahead,
)
from .breadth import (
    check_breadth_control,
    check_breadth_vs_enb,
    check_residual_breadth,
    effective_names,
    enb,
    residual_breadth,
    residual_breadth_panel,
)
from .report import BLOCK, OK, WARN, AuditReport, Finding
from .significance import (
    check_deflated_sharpe,
    check_drawdown,
    check_nw_lag_sensitivity,
    check_year_concentration,
    deflated_sharpe,
    newey_west_t,
    to_returns,
)
from .turnover_cost import (
    breakeven_cost,
    check_breakeven,
    check_gross_net_reconcile,
    check_implied_turnover,
    check_turnover_basis,
)

__version__ = "0.1.0"

__all__ = [
    "audit",
    "audit_strategy",
    "AuditReport",
    "Finding",
    # 识别与能力
    "detect_frame",
    "detect_all",
    "classify_series",
    "Detected",
    "CHECKS",
    "available",
    "missing_value",
    "matrix_text",
    # 族三
    "check_year_concentration",
    "check_nw_lag_sensitivity",
    "check_deflated_sharpe",
    "check_drawdown",
    "deflated_sharpe",
    "newey_west_t",
    "to_returns",
    "BLOCK",
    "WARN",
    "OK",
    # 契约
    "load_weights",
    "load_prices",
    "check_gross",
    "to_matrix",
    "normalize_gross",
    # 核心数学
    "period_returns",
    "daily_path",
    "turnover",
    "drift_weights",
    "rank_autocorr_turnover",
    "price_matrix",
    "align",
    "annualize",
    "periods_per_year",
    "MISSING_POLICIES",
    # 族一
    "check_turnover_basis",
    "check_implied_turnover",
    "check_breakeven",
    "check_gross_net_reconcile",
    "breakeven_cost",
    # 族二
    "check_rebalance_alignment",
    "check_weight_lookahead",
    "check_universe_survivorship",
    "check_membership_accounting",
    # 族四
    "check_residual_breadth",
    "check_breadth_control",
    "check_breadth_vs_enb",
    "residual_breadth_panel",
    "residual_breadth",
    "enb",
    "effective_names",
    "__version__",
]
