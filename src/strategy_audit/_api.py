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
from . import capacity as cp
from . import lookahead as la
from . import navquality as nq
from . import prescribe as px
from . import significance as sg
from . import turnover_cost as tc
from . import parameter_audit as pa
from .contract import (check_gross, check_nav_reconciliation, load_prices,
                       load_weights, normalize_gross, to_matrix)
from .core import period_returns, periods_per_year, price_matrix
from .core import turnover as core_turnover
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


def _warn_daily_gap(index, rep: AuditReport) -> None:
    """日频输入有异常缺口时，禁止把缺失观察静默当作较低频率。

    不能直接看日历天：A 股春节/国庆前后的正常相邻交易日可相隔 11 天。
    改看中间跳过的工作日；节假日最多造成少量工作日缺口，下载遗漏数周或
    数月则会显著更大。这里仍只是数据完整性提示，不假装有交易所日历。
    """
    d = pd.DatetimeIndex(pd.to_datetime(pd.Index(index))).sort_values()
    if len(d) < 3:
        return
    gaps = pd.Series(d).diff().dt.days.dropna()
    if gaps.empty or float(gaps.median()) > 4:
        return
    skipped = [(len(pd.bdate_range(left, right)) - 2, left, right)
               for left, right in zip(d[:-1], d[1:])]
    longest, left, right = max(skipped, key=lambda x: x[0])
    if longest > 10:
        rep.add(WARN, "日频序列存在长缺口",
                f"{left.date()} ~ {right.date()} 之间跳过 {longest} 个工作日",
                "年化按已观测日期估算；这不是把缺失期间当作零收益。"
                "请确认该缺口是正常停市还是行情/净值下载不完整，"
                "否则年化收益、波动和 Sharpe 都不应直接比较",
                section="输入识别")


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
          name: str = "策略审计", show_detection: bool = True,
          execution: dict | None = None, params: dict | None = None,
          signals=None, signals_alt=None, delisted=None, trades=None) -> AuditReport:
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
    execution = {"entry": "close", "exit": "close", "signal_lag": 0,
                 **(execution or {})}
    if execution["entry"] not in ("close", "open") or execution["exit"] not in ("close", "open"):
        raise ValueError("execution.entry/exit 必须是 close 或 open")
    if int(execution["signal_lag"]) < 0:
        raise ValueError("execution.signal_lag 必须为非负整数")
    params = dict(params or {})
    if trades is not None:
        trades = trades.copy() if isinstance(trades, pd.DataFrame) else _read_path(str(trades))
    sig_obj = _read_path(str(signals)) if isinstance(signals, (str, Path)) else signals
    alt_obj = _read_path(str(signals_alt)) if isinstance(signals_alt, (str, Path)) else signals_alt
    sig = pa.load_signals(sig_obj) if sig_obj is not None else None
    sig_alt = pa.load_signals(alt_obj) if alt_obj is not None else None
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
    price_fields = {}
    if weights is not None:
        w = load_weights(weights, rep)
        if w is not None:
            w = check_gross(w, rep)
            wm = normalize_gross(to_matrix(w))
    if prices is not None:
        p = load_prices(prices, rep)
        if p is not None:
            pm = price_matrix(p)
            price_fields["close"] = pm
            for field in ("open", "up_limit", "down_limit", "is_suspended", "is_st"):
                if field in p.columns:
                    price_fields[field] = price_matrix(p, field)

    # 权重表自带 close 时，可以直接当价格面板用
    if pm is None and weights is not None and "close" in weights.columns:
        p = load_prices(weights[["date", "code", "close"]].dropna(), rep)
        if p is not None:
            pm = price_matrix(p)
            price_fields["close"] = pm
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
    if "open" in price_fields:
        have.add(cap.OPEN)
    if any(k in price_fields for k in ("up_limit", "down_limit", "is_suspended", "is_st")):
        have.add(cap.FLAGS)
    if sig is not None:
        have.add(cap.SIG)
    if sig_alt is not None:
        have.add(cap.SIG_ALT)
    if params:
        have.add(cap.PARAMS)
    if delisted is not None:
        have.add(cap.DELISTED)
    # ★ 只有带 exit_date 的交易明细才算 TRADES：跌停出场顺延需要出场日，
    # 只有 entry_date/code 的表定位不到出场，报「能审」就是把没查显示成查过。
    if trades is not None and "exit_date" in getattr(trades, "columns", ()):
        have.add(cap.TRADES)

    # 成交额列（族五容量用）。识别层会把 amount 一起带进价格表。
    am = cp.amount_matrix(prices) if prices is not None else None
    if am is not None and am.notna().any().any():
        have.add(cap.AMT)
    else:
        am = None

    # 有权重+价格就能自己算出收益曲线 ⇒ 族三也能跑
    rets = None
    ppy = None
    if wm is not None and pm is not None and len(wm.index) >= 3:
        entry_pm = price_fields.get(execution["entry"])
        exit_pm = price_fields.get(execution["exit"])
        if entry_pm is None:
            rep.skip("执行口径收益", f"entry={execution['entry']} 但价格面板没有该列；不回退到 close")
        elif exit_pm is None:
            rep.skip("执行口径收益", f"exit={execution['exit']} 但价格面板没有该列")
        else:
            pr = period_returns(wm, pm, entry_pm=entry_pm, exit_pm=exit_pm)
            rets = pr["ret"]
            ppy = periods_per_year(wm.index)
        _warn_daily_gap(pm.index, rep)
        if rets is not None:
            have.add(cap.NAV)
        # ★ 同时给了自报净值时【必须对账】，不能静默丢弃。
        # 第一版这里是 elif：权重+价格在手就自己重算，客户交上来的那条
        # 曲线连看一眼都没有。盲测实测两条差 6.0%（自报 6.19% / 重算 6.66%，
        # 原因是月内日频再平衡，权重表里看不出来）—— 于是所有检查
        # 都算在一条【客户从未汇报过】的净值上，而报告照样打印得像模像样。
        # 本工具宣称审的就是「回测到净值这一段」，那这一项就是它的本职。
        if nav is not None:
            own, own_role = nav
            have.add(cap.OWN_NAV)
            # 换手在这里先算一次传进去：良性方向（自报低于重算）要把缺口
            # 折算成隐含成本才能判断「像不像扣了成本」。族一稍后会再算一次
            # 并报口径对比 —— 这里只借用数值，不出结论。
            try:
                _to = float(core_turnover(wm, pm)["drift_adj"].mean())
            except Exception:
                _to = None
            if rets is not None and cap.can_run("nav_recon", have):
                with rep.check("nav_recon"):
                    check_nav_reconciliation(rets, _clean_series(own), own_role, rep,
                                             turnover=_to)
    if rets is None and nav is not None:
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
        _warn_daily_gap(rets.index, rep)
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
            for key, fn in (
                    ("day0", la.check_rebalance_alignment),
                    ("w_look", la.check_weight_lookahead),
                    ("univ", la.check_universe_survivorship),
                    ("member", la.check_membership_accounting)):
                if cap.can_run(key, have):
                    with rep.check(key):
                        if key == "univ":
                            fn(wm, pm, rep, delisted=delisted)
                        else:
                            fn(wm, pm, rep)

    # ---- 族一：换手与成本 ----
    if wm is not None and len(wm.index) >= 3:
        if cap.can_run("to_basis", have):
            with rep.check("to_basis"):
                to = tc.check_turnover_basis(wm, pm, rep)
        else:
            to = None
        if to is None:
            from .core import turnover
            to = turnover(wm, None)
        if cap.can_run("to_implied", have):
            with rep.check("to_implied"):
                tc.check_implied_turnover(wm, to, rep)
        if cap.can_run("breakeven", have) and rets is not None and ppy:
            with rep.check("breakeven"):
                tc.check_breakeven(rets, to, ppy, rep, bench=bench)
        if cap.can_run("reconcile", have) and rets is not None and net is not None:
            with rep.check("reconcile"):
                tc.check_gross_net_reconcile(rets, net, to, rep)

    # ---- 族七：净值质量（只要曲线）----
    # ★ 排在族三之前：族三审「这个数字可不可信」，但它默认曲线本身是真的；
    # 族七审的正是那个前提。曲线被平滑过时，再精确的显著性检验
    # 也只是把一个被低估的波动算得更精确。
    if rets is not None and len(rets) >= nq.MIN_N:
        if cap.can_run("smoothing", have):
            with rep.check("smoothing"):
                nq.check_smoothing(rets, ppy or 252.0, rep)
        # 停滞估值要看【原始序列】：自报净值时用净值本身（相邻点是否相等），
        # 只有收益率时退回收益率。重算出来的收益曲线传 role="ret"。
        if cap.can_run("stale_nav", have):
            with rep.check("stale_nav"):
                if nav is not None:
                    _own, _role = nav
                    nq.check_stale(_clean_series(_own), _role, rep)
                else:
                    nq.check_stale(rets, "ret", rep)
        if cap.can_run("dressing", have):
            with rep.check("dressing"):
                nq.check_period_end_dressing(rets, rep)
    elif rets is not None:
        rep.skip("净值质量（全部 3 项）",
                 f"序列只有 {len(rets)} 个点（需要 ≥{nq.MIN_N}）")

    # ---- 族三：策略层显著性（只要曲线）----
    if rets is not None and len(rets) >= 6:
        for key, fn in (
                ("year_conc", lambda: sg.check_year_concentration(rets, rep, bench=bench)),
                ("nw_lag", lambda: sg.check_nw_lag_sensitivity(rets, rep)),
                ("dsr", lambda: sg.check_deflated_sharpe(rets, ppy or 252.0, rep,
                                                          n_trials=n_trials)),
                ("drawdown", lambda: sg.check_drawdown(rets, rep))):
            if cap.can_run(key, have):
                with rep.check(key):
                    fn()
    elif rets is not None:
        rep.skip("策略层显著性（全部 4 项）",
                 f"有效收益只有 {len(rets)} 期，至少需要 6 期")

    # ---- 族四：风险身份（放最后：它不问净值对不对，问风险预算编得对不对）----
    if wm is not None and pm is not None and len(wm.index) >= 3:
        d, notes = br.residual_breadth_panel(wm, pm)
        for key, fn in (
                ("breadth", lambda: br.check_residual_breadth(d, rep, notes)),
                ("breadth_ctrl", lambda: br.check_breadth_control(d, rep)),
                ("breadth_enb", lambda: br.check_breadth_vs_enb(d, rep))):
            if cap.can_run(key, have):
                with rep.check(key):
                    fn()

    # ---- 族五：容量与可成交性（这些收益你到底拿不拿得到）----
    if wm is not None and pm is not None and len(wm.index) >= 3:
        if cap.can_run("untradable", have):
            with rep.check("untradable"):
                st = price_fields.get("is_st")
                flags = {k: price_fields[k] for k in ("up_limit", "down_limit", "is_suspended")
                         if k in price_fields}
                cp.check_untradable(wm, pm, rep, st=st,
                                    flags=flags or None)
        if cap.can_run("capacity", have):
            with rep.check("capacity"):
                cp.check_capacity(wm, am, rep, pm=pm)
        if cap.can_run("liq_tilt", have):
            with rep.check("liq_tilt"):
                cp.check_liquidity_tilt(wm, am, pm, rep)
        if cap.can_run("size_decay", have) and rets is not None and ppy:
            with rep.check("size_decay"):
                cp.check_size_decay(rets, wm, am, ppy, rep)

    # ---- 族六：处方层（放最末：只有前五族问过「可不可信」之后，
    # 才轮到「怎么改」。给一份不可信的净值出处方是没有意义的）----
    if wm is not None and pm is not None and len(wm.index) >= 3:
        if cap.can_run("to_value", have):
            with rep.check("to_value"):
                px.check_turnover_value(wm, pm, rep)
        if cap.can_run("to_split", have):
            with rep.check("to_split"):
                px.check_turnover_split(wm, pm, rep)
        if cap.can_run("prescribe", have) and ppy:
            with rep.check("prescribe"):
                px.check_prescription(wm, pm, ppy, rep)

    # ---- 族七：参数邻域与信号自洽（只体检，不寻优）----
    if wm is not None and sig is not None and cap.can_run("signal_consistency", have):
        with rep.check("signal_consistency"):
            pa.check_signal_consistency(wm, sig, int(execution["signal_lag"]),
                                        (pm.index if pm is not None else wm.index), rep)
    if wm is not None and pm is not None:
        if cap.can_run("holding_neighborhood", have):
            with rep.check("holding_neighborhood"):
                pa.check_holding(wm, price_fields.get("close", pm),
                                 price_fields.get(execution["entry"], pm),
                                 params.get("holding_days"), rep, trades=trades)
        if cap.can_run("topn_neighborhood", have):
            with rep.check("topn_neighborhood"):
                pa.check_topn(sig, price_fields.get("close", pm),
                              price_fields.get(execution["entry"], pm),
                              params.get("holding_days"), params.get("top_n"), rep)
        if cap.can_run("entry_neighborhood", have):
            with rep.check("entry_neighborhood"):
                pa.check_entry(wm, price_fields["close"], price_fields["open"], rep)
        if cap.can_run("vendor_pair", have):
            with rep.check("vendor_pair"):
                pa.check_vendor_pair(sig, sig_alt, price_fields["close"],
                                     price_fields.get(execution["entry"], pm), params, rep)
        if cap.can_run("deferred_exit", have):
            with rep.check("deferred_exit"):
                pa.check_deferred_exit(
                    trades, price_fields.get("close", pm),
                    {k: price_fields[k] for k in ("down_limit",) if k in price_fields},
                    rep, wm=wm,
                    entry_pm=price_fields.get(execution["entry"], pm))

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
        if c.key == "vendor_pair" and cap.SIG_ALT not in have:
            rep.skip(c.name, "缺第二家供应商的信号面板；补上可回答「alpha 是策略的还是数据商口径的」——这是本工具最强的单条结论")
        else:
            lack = "、".join(cap.label(k) for k in c.needs if k not in have)
            rep.skip(c.name, f"缺{lack}")

    return rep


def audit_strategy(weights, prices, *, net_returns=None, benchmark=None,
                   name: str = "策略审计", execution=None, params=None,
                   signals=None, signals_alt=None, delisted=None, trades=None) -> AuditReport:
    """旧接口，保留向后兼容。新代码请用 audit()。"""
    return audit(weights, prices, net=net_returns, benchmark=benchmark,
                 name=name, execution=execution, params=params,
                 signals=signals, signals_alt=signals_alt, delisted=delisted,
                 trades=trades)
