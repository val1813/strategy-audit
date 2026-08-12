"""顶层 API。

    from strategy_audit import audit

    rep = audit(weights, prices)      # 想给什么给什么，顺序无关
    rep = audit(nav)                  # 只有净值也能审
    rep = audit("持仓.csv", "行情.parquet")
    print(rep.text())

★ 设计原则：你给什么，我就审什么
------------------------------
不要求固定的参数位置、不要求列名、不要求你先整理数据。
自动识别（detect.py）+ 能力矩阵（capability.py）负责把
「能审什么、审不了什么、缺什么补上能多审几项」摆在报告最前面。

审不了 ≠ 审过通过 —— 这条边界必须在报告开头就画清楚。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import capability as cap
from . import lookahead as la
from . import significance as sg
from . import turnover_cost as tc
from .contract import check_gross, load_prices, load_weights, normalize_gross, to_matrix
from .core import period_returns, periods_per_year, price_matrix
from .detect import Detected, detect_all, detect_frame
from .report import BLOCK, WARN, AuditReport


def _read_path(p: str) -> pd.DataFrame:
    f = Path(p)
    if not f.exists():
        raise FileNotFoundError(f"文件不存在：{f}")
    if f.suffix.lower() in (".parquet", ".pq"):
        return pd.read_parquet(f)
    if f.suffix.lower() in (".xlsx", ".xls"):
        return pd.read_excel(f)
    return pd.read_csv(f)


def _as_series(o) -> pd.Series | None:
    """把显式传入的 net/benchmark 规整成 Series（接受路径/表/序列）。"""
    if o is None:
        return None
    if isinstance(o, pd.Series):
        return o.dropna().astype(float).sort_index()
    if isinstance(o, (str, Path)):
        o = _read_path(str(o))
    if isinstance(o, pd.DataFrame):
        d = detect_frame(o)
        if d.series is not None:
            return d.series
        raise ValueError(
            f"net/benchmark 需要单列时间序列（date + 一列数值），"
            f"实际认成了 {d.kind}：{'；'.join(d.notes)}")
    raise TypeError(f"net/benchmark 不支持的类型 {type(o).__name__}")


def _prepare(inputs) -> list[Detected]:
    """把混杂输入（路径/DataFrame/Series/(name,obj)）统一识别。"""
    objs = []
    for o in inputs:
        if o is None:
            continue
        if isinstance(o, (str, Path)):
            objs.append((str(o), _read_path(str(o))))
        else:
            objs.append(o)
    return detect_all(objs)


def audit(*inputs, net=None, benchmark=None, n_trials: int = 1,
          name: str = "策略审计", show_detection: bool = True) -> AuditReport:
    """审一份回测。参数随便给，顺序无关。

    inputs     权重表 / 价格表 / 净值或收益率序列 / 文件路径，任意组合
    net        你自己算的【净】收益序列（用于毛净对账）
    benchmark  基准收益序列（盈亏平衡成本按超额算）
    n_trials   你试过多少个配置才选出这一个（用于多重检验折扣）

    ★ net / benchmark 必须用关键字传，不能混在 inputs 里。
    原因：一条收益率序列从【数值上】无法区分它是策略净收益、
    基准、还是策略毛收益 —— 三者量级、正负、自相关都一样。
    猜错会让毛净对账算在错的两条曲线上，而报告照样打印得像模像样。
    这是少数几个「宁可要求用户明说」的地方。
    """
    rep = AuditReport(title=name)
    det = _prepare(inputs)

    if not det:
        rep.add(BLOCK, "没有输入", "至少给一个：权重表、价格表，或一条净值曲线",
                "最低门槛是一条净值曲线（date + 一列数值），"
                "那能审 4 项；权重表 + 价格表能审全部 12 项",
                section="输入识别")
        return rep

    # ---- 归集识别结果 ----
    weights = prices = None
    nav = None
    # 显式传入的角色优先，识别只能补空缺
    net = _as_series(net)
    bench = _as_series(benchmark)
    for d in det:
        if d.kind == "weights" and weights is None:
            weights = d.frame
        elif d.kind == "prices" and prices is None:
            prices = d.frame
        elif d.kind == "series":
            if d.role in ("nav", "ret") and nav is None:
                nav = (d.series, d.role)
            elif d.role == "net" and net is None:
                net = d.series
            elif d.role == "bench" and bench is None:
                bench = d.series

    if show_detection:
        lines = []
        for d in det:
            kind = {"weights": "权重面板", "prices": "价格面板",
                    "series": f"序列（{d.role}）",
                    "unknown": "★ 认不出来"}.get(d.kind, d.kind)
            lines.append(f"{d.source} → {kind}")
            for n in d.notes:
                lines.append(f"  · {n}")
        rep.add("OK" if all(d.kind != "unknown" for d in det) else WARN,
                "输入识别结果", "\n".join(lines),
                "★ 请核对上面的识别是否正确。认错了会让所有检查算在错的"
                "东西上，而报告照样打印得像模像样",
                section="输入识别")

    # ---- 契约检查 ----
    wm = pm = None
    if weights is not None:
        w = load_weights(weights, rep)
        if w is not None:
            w = check_gross(w, rep)
            wm = normalize_gross(to_matrix(w))
    if prices is not None:
        p = load_prices(prices, rep)
        if p is not None:
            pm = price_matrix(p)

    # 权重表自带 close 时，可以直接当价格面板用
    if pm is None and weights is not None and "close" in weights.columns:
        p = load_prices(weights[["date", "code", "close"]].dropna(), rep)
        if p is not None:
            pm = price_matrix(p)
            rep.skip("价格面板", "未单独提供，已从权重表的 close 列取用")

    have = set()
    if wm is not None:
        have.add(cap.W)
    if pm is not None:
        have.add(cap.P)
    if net is not None:
        have.add(cap.NET)
    if bench is not None:
        have.add(cap.BENCH)

    # 有权重+价格就能自己算出收益曲线 ⇒ 族三也能跑
    rets = None
    ppy = None
    if wm is not None and pm is not None and len(wm.index) >= 3:
        pr = period_returns(wm, pm)
        rets = pr["ret"]
        ppy = periods_per_year(wm.index)
        have.add(cap.NAV)
    elif nav is not None:
        s, role = nav
        rets = sg.to_returns(s, role)
        ppy = periods_per_year(rets.index) if len(rets) > 2 else 252.0
        have.add(cap.NAV)

    rep.stats["capability"] = sorted(have)
    rep.capability = cap.matrix_text(have)

    if wm is not None:
        rep.stats.update(
            n_dates=len(wm.index),
            n_names=int((wm != 0).any().sum()),
            span=f"{wm.index.min().date()} ~ {wm.index.max().date()}")
    if ppy:
        rep.stats["periods_per_year"] = ppy
    elif rets is not None and len(rets):
        rep.stats.update(
            n_dates=len(rets),
            span=f"{rets.index.min().date()} ~ {rets.index.max().date()}")

    if not have:
        rep.add(BLOCK, "没有可用输入",
                "所有输入都没能识别成权重表/价格表/净值曲线",
                "看上面的识别结果。最低要求：一张有 date 列和一列数值的表",
                section="输入识别")
        return rep

    # ---- 族二：前视与记账（先跑：先问净值是不是真的）----
    if wm is not None and pm is not None:
        if len(wm.index) < 3:
            rep.add(BLOCK, "调仓期数不足", f"仅 {len(wm.index)} 个调仓日",
                    "至少 3 期才能算换手；多数检查需要 6~7 期",
                    section="输入契约")
        else:
            la.check_rebalance_alignment(wm, pm, rep)
            la.check_weight_lookahead(wm, pm, rep)
            la.check_universe_survivorship(wm, pm, rep)
            la.check_membership_accounting(wm, pm, rep)

    # ---- 族一：换手与成本 ----
    if wm is not None and len(wm.index) >= 3:
        to = tc.check_turnover_basis(wm, pm, rep) if pm is not None else None
        if to is None:
            from .core import turnover
            to = turnover(wm, None)
        tc.check_implied_turnover(wm, to, rep)
        if rets is not None and ppy:
            tc.check_breakeven(rets, to, ppy, rep, bench=bench)
            if net is not None:
                tc.check_gross_net_reconcile(rets, net, to, rep)

    # ---- 族三：策略层显著性（只要曲线）----
    if rets is not None and len(rets) >= 6:
        sg.check_year_concentration(rets, rep, bench=bench)
        sg.check_nw_lag_sensitivity(rets, rep)
        sg.check_deflated_sharpe(rets, ppy or 252.0, rep, n_trials=n_trials)
        sg.check_drawdown(rets, rep)

    # ---- 把审不了的登记成「未能检查」----
    # ★ 缺什么要说人话，不能漏内部键名（"缺 net" 用户看不懂）
    ok, no = cap.available(have)
    for c in no:
        lack = "、".join(cap.label(k) for k in c.needs if k not in have)
        rep.skip(c.name, f"缺{lack}")

    return rep


def audit_strategy(weights, prices, *, net_returns=None, benchmark=None,
                   name: str = "策略审计") -> AuditReport:
    """旧接口，保留向后兼容。新代码请用 audit()。"""
    return audit(weights, prices, net=net_returns, benchmark=benchmark,
                 name=name)
