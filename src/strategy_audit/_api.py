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

from . import breadth as br
from . import capability as cap
from . import lookahead as la
from . import significance as sg
from . import turnover_cost as tc
from .contract import check_gross, load_prices, load_weights, normalize_gross, to_matrix
from .core import period_returns, periods_per_year, price_matrix
from .detect import Detected, detect_all, detect_frame
from .report import BLOCK, WARN, AuditReport


def _read_path(p: str) -> pd.DataFrame:
    """读文件。缺可选依赖时给【能照做的】提示，而不是抛 ImportError。

    ★ 我们声称支持 parquet/xlsx，但 pyarrow/openpyxl 都是可选依赖
    （不想为了读一种格式把体积压给所有用户）。那就必须自己接住
    缺依赖的情况 —— 否则用户看到的是 pandas 抛的一句
    "Missing optional dependency 'openpyxl'"，还得自己去猜装什么。
    """
    f = Path(p)
    if not f.exists():
        raise FileNotFoundError(f"文件不存在：{f}")
    suf = f.suffix.lower()
    try:
        if suf in (".parquet", ".pq"):
            return pd.read_parquet(f)
        if suf in (".xlsx", ".xls", ".xlsm"):
            return pd.read_excel(f)
        if suf in (".json",):
            return pd.read_json(f)
        return pd.read_csv(f)
    except ImportError as e:
        pkg = ("pyarrow" if suf in (".parquet", ".pq")
               else "openpyxl" if suf in (".xlsx", ".xlsm")
               else "xlrd" if suf == ".xls" else None)
        if pkg:
            raise SystemExit(
                f"读 {f.name} 需要 {pkg}，当前环境没装。\n"
                f"  装它：pip install {pkg}\n"
                f"  或者把文件导出成 csv 再传进来（csv 不需要额外依赖）"
            ) from e
        raise
    except Exception as e:
        raise SystemExit(
            f"读不了 {f.name}（{type(e).__name__}：{e}）。\n"
            f"  支持 csv / parquet / xlsx / json；"
            f"若是编码问题，请另存为 UTF-8 的 csv"
        ) from e


def _clean_series(s: pd.Series) -> pd.Series:
    """去掉 NaN/±inf，按日期排序，重复日期保留最后一条。

    ★ ±inf 必须显式清掉。实测：净值里一个 inf 会让 pct_change 产生
    inf/nan，再进 NW t 的方差计算就是 `invalid value in scalar divide`——
    工具直接崩，而用户完全不知道是哪一行数据的问题。
    """
    v = pd.Series(s).astype(float)
    v = v[np.isfinite(v.values)]
    if len(v) and not v.index.is_monotonic_increasing:
        v = v.sort_index()
    if len(v) and v.index.has_duplicates:
        v = v[~v.index.duplicated(keep="last")]
    return v


def _as_series(o) -> pd.Series | None:
    """把显式传入的 net/benchmark 规整成 Series（接受路径/表/序列）。"""
    if o is None:
        return None
    if isinstance(o, pd.Series):
        return _clean_series(o)
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
                f"那能审 {len(cap.available({cap.NAV})[0])} 项；"
                f"权重表 + 价格表能审 "
                f"{len(cap.available({cap.W, cap.P, cap.NAV})[0])} 项",
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
        n_raw = len(s)
        s = _clean_series(s)
        if len(s) < n_raw:
            rep.add(WARN, "序列里有无效值",
                    f"丢弃 {n_raw - len(s)} 个 NaN/±inf/重复日期的点"
                    f"（原 {n_raw} 点，剩 {len(s)} 点）",
                    "±inf 通常来自除以 0 的收益计算或缺价日的除权处理。"
                    "请核对原始数据，不要指望丢弃它们就没事了",
                    section="输入识别")
        rets = _clean_series(sg.to_returns(s, role))
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
                "最低可用输入是一条【至少 7 个点】的净值曲线"
                "（date + 一列数值）；权重表 + 价格表能审 "
                f"{len(cap.available({cap.W, cap.P, cap.NAV})[0])} 项。"
                "想先看报告长什么样，跑 strategy-audit --demo",
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
    elif rets is not None:
        rep.skip("策略层显著性（全部 4 项）",
                 f"有效收益只有 {len(rets)} 期，至少需要 6 期")

    # ---- 族四：风险身份（放最后：它不问净值对不对，问风险预算编得对不对）----
    if wm is not None and pm is not None and len(wm.index) >= 3:
        d, notes = br.residual_breadth_panel(wm, pm)
        br.check_residual_breadth(d, rep, notes)
        br.check_breadth_control(d, rep)
        br.check_breadth_vs_enb(d, rep)

    # ★ 一条 finding 都没有时必须解释为什么，不能交一份空报告。
    # 实测：单行表/全 NaN/净值全 0 这三种输入会走到这里 ——
    # 用户看到的是「什么都没说」，比报错更让人无从下手。
    if not any(f.section not in ("输入识别", "输入契约") for f in rep.findings):
        why = []
        if rets is None:
            why.append("没能从输入里得到收益序列")
        elif len(rets) < 6:
            why.append(f"有效收益只有 {len(rets)} 期（需要 ≥6）")
        if wm is None:
            why.append("没有权重面板")
        if pm is None:
            why.append("没有价格面板")
        rep.add(BLOCK, "没有任何检查能跑起来",
                "；".join(why) or "输入不足",
                "上面的「未能检查」列出了每一项缺什么。"
                "最低可用输入是一条【至少 7 个点】的净值曲线"
                "（date + 一列数值）；想看报告长什么样先跑 --demo",
                section="输入识别")

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
