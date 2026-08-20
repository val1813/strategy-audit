"""Offline parameter-neighborhood and signal consistency checks.

These checks diagnose a declared setting. They never optimize or recommend a
replacement setting.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .report import OK, SKIP, WARN, AuditReport
from .core import periods_per_year

SECTION = "参数邻域体检"
H_GRID = tuple(range(1, 11))
TOPN_GRID = (1, 3, 5, 10, 20, 50)


def load_signals(obj) -> pd.DataFrame | None:
    if obj is None:
        return None
    d = obj.copy() if isinstance(obj, pd.DataFrame) else pd.read_csv(obj)
    aliases = {"日期": "date", "交易日": "date", "代码": "code",
               "证券代码": "code", "股票代码": "code", "信号": "score",
               "信号值": "score", "得分": "score"}
    d = d.rename(columns={c: aliases.get(str(c), c) for c in d.columns})
    if not {"date", "code", "score"} <= set(d.columns):
        raise ValueError("signals 需要 date/code/score 三列")
    d = d[["date", "code", "score"]].copy()
    d["date"] = pd.to_datetime(d["date"])
    d["code"] = d["code"].astype(str)
    d["score"] = pd.to_numeric(d["score"], errors="coerce")
    if d.duplicated(["date", "code"]).any():
        raise ValueError("signals 存在重复 (date, code)")
    return d.dropna(subset=["date", "code", "score"]).sort_values(["date", "code"])


def _paired(a, b) -> dict:
    aa, bb = pd.Series(a), pd.Series(b)
    if aa.index.is_unique and bb.index.is_unique:
        pair = pd.concat([aa.rename("a"), bb.rename("b")], axis=1, join="inner").dropna()
        # The split-half gate is a time-stability check.  Never let the
        # caller's row order (e.g. a trade export sorted by return) define
        # “first half” and “second half”.  Trade-level returns carry entry
        # dates in their first MultiIndex level; ordinary DatetimeIndex is
        # handled as well.  Stable sorting preserves deterministic ties.
        if isinstance(pair.index, pd.MultiIndex):
            first = pair.index.get_level_values(0)
            if isinstance(first, pd.DatetimeIndex):
                pair = pair.sort_index(level=0, kind="stable")
        elif isinstance(pair.index, pd.DatetimeIndex):
            pair = pair.sort_index(kind="stable")
        d = pair["a"] - pair["b"]
    else:
        d = aa.reset_index(drop=True) - bb.reset_index(drop=True)
    d = d[np.isfinite(d)]
    n = len(d)
    mean = float(d.mean()) if n else np.nan
    sd = float(d.std(ddof=1)) if n > 1 else np.nan
    se = sd / np.sqrt(n) if n > 1 and sd > 0 else np.nan
    t = mean / se if np.isfinite(se) and se > 0 else np.nan
    mde = 1.96 * se if np.isfinite(se) else np.nan
    half = n // 2
    same = (n >= 4 and np.sign(d.iloc[:half].mean()) == np.sign(d.iloc[half:].mean())
            and np.sign(mean) != 0)
    passed = bool(n >= 100 and np.isfinite(t) and abs(t) >= 2 and same
                  and np.isfinite(mde) and abs(mean) >= mde)
    return dict(n=n, mean=mean, t=t, mde=mde, halves_same=same, passed=passed)


def _trade_events(wm: pd.DataFrame, trades: pd.DataFrame | None = None) -> pd.DataFrame:
    """Return trade-level entry events, never rebalance-level portfolios.

    An explicit trade table wins when supplied. Otherwise a holding segment is
    one event at each 0 -> positive transition in the weight matrix. This
    preserves the event identity across H comparisons and avoids cash dilution.
    """
    if trades is not None:
        required = {"entry_date", "code"}
        if not required <= set(trades.columns):
            raise ValueError("trades 需要 entry_date/code 两列")
        out = trades.copy()
        out["entry_date"] = pd.to_datetime(out["entry_date"])
        out["code"] = out["code"].map(lambda x: str(x) if not isinstance(x, str) else x)
        if "exit_date" in out.columns:
            out["exit_date"] = pd.to_datetime(out["exit_date"])
        return out.reset_index(drop=True)
    x = wm.sort_index().copy()
    prev = x.shift(1).fillna(0.0)
    rows = []
    for d in x.index:
        entered = (x.loc[d] > 0) & (prev.loc[d] <= 0)
        rows.extend((d, str(c)) for c in x.columns[entered])
    return pd.DataFrame(rows, columns=["entry_date", "code"])


def _event_returns(wm, entry_pm, exit_pm, h: int,
                   trades: pd.DataFrame | None = None,
                   use_actual_exit: bool = False) -> pd.Series:
    """Trade-level returns where H counts the entry day as day 1.

    Thus H=1 exits on the entry session and H=3 exits at the third observed
    trading day (entry position + 2), matching a trade table's entry/exit
    convention. ``use_actual_exit`` is used only for the declared baseline
    when an explicit exit_date is supplied.
    """
    days = list(exit_pm.index)
    pos = {d: i for i, d in enumerate(days)}
    events = _trade_events(wm, trades)
    out = {}
    for event_id, row in events.iterrows():
        d, code = row["entry_date"], row["code"]
        if d not in pos or d not in entry_pm.index:
            continue
        if use_actual_exit and "exit_date" in row and pd.notna(row["exit_date"]):
            end = pd.Timestamp(row["exit_date"])
            if end not in pos:
                continue
        else:
            if h < 1 or pos[d] + h - 1 >= len(days):
                continue
            end = days[pos[d] + h - 1]
        code_key = code if code in entry_pm.columns else (
            str(code) if str(code) in entry_pm.columns else code)
        if code_key not in entry_pm.columns or code_key not in exit_pm.columns:
            continue
        a, b = entry_pm.loc[d, code_key], exit_pm.loc[end, code_key]
        if not np.isfinite(a) or not np.isfinite(b) or a == 0:
            continue
        out[(pd.Timestamp(d), int(event_id))] = float(b / a - 1.0)
    if not out:
        return pd.Series(dtype=float)
    idx = pd.MultiIndex.from_tuples(out, names=["entry_date", "event_id"])
    return pd.Series(list(out.values()), index=idx, dtype=float)


def check_holding(wm, close_pm, entry_pm, current: int, rep: AuditReport,
                  trades: pd.DataFrame | None = None) -> None:
    if not isinstance(current, int) or current < 1:
        rep.skip("持有期邻域体检", "params.holding_days 必须是正整数")
        return
    base = _event_returns(wm, entry_pm, close_pm, current, trades,
                          use_actual_exit=trades is not None and "exit_date" in trades)
    rep.stats["holding_trade_events"] = int(len(base))
    rows = []
    for h in H_GRID:
        if h == current:
            continue
        alt = _event_returns(wm, entry_pm, close_pm, h, trades)
        z = _paired(base, alt)
        rows.append((h, z))
    if not rows:
        rep.skip("持有期邻域体检", "没有足够的入场日与后续价格")
        return
    parts = [f"H={current} vs H={h}: t={z['t']:+.2f}, n={z['n']}, "
             f"MDE={z['mde']:.2%}, {'实质差异' if z['passed'] else '不可区分'}"
             for h, z in rows]
    shorter = [z for h, z in rows if h < current and z["passed"] and z["mean"] > 0]
    longer = [z for h, z in rows if h > current]
    platform = bool(shorter and longer and all(not z["passed"] for z in longer))
    conclusion = (f"H≥{current} 呈平台而非孤立尖峰。" if platform else
                  "当前档位与邻域的差异见逐档配对结果。")
    rep.add(OK, "持有期邻域体检",
            "H 按含入场日计数（H=1 为入场日）\n" + conclusion + "\n" + "\n".join(parts),
            "这是参数体检，不据此更换参数档位", section=SECTION)


def _rank_returns(signals, pm, h, n_top, entry_pm=None) -> pd.Series:
    entry_pm = pm if entry_pm is None else entry_pm
    days = list(pm.index); pos = {d: i for i, d in enumerate(days)}; out = []
    for d, g in signals.groupby("date"):
        if d not in pos or h < 1 or pos[d] + h - 1 >= len(days) or d not in entry_pm.index:
            continue
        chosen = g.sort_values(["score", "code"], ascending=[False, True]).head(n_top)["code"]
        a = entry_pm.loc[d].reindex(chosen); b = pm.loc[days[pos[d] + h - 1]].reindex(chosen)
        rr = (b / a - 1).replace([np.inf, -np.inf], np.nan).dropna()
        if len(rr): out.append((d, float(rr.mean())))
    return pd.Series(dict(out), dtype=float).sort_index()


def check_topn(signals, close_pm, entry_pm, holding_days, current, rep):
    if not isinstance(current, int) or current < 1:
        rep.skip("Top-N 邻域体检", "params.top_n 必须是正整数")
        return
    h = int(holding_days) if isinstance(holding_days, int) and holding_days > 0 else 1
    lines = []
    for n in TOPN_GRID:
        r = _rank_returns(signals, close_pm, h, n, entry_pm)
        ppy = periods_per_year(r.index) if len(r) > 2 else np.nan
        ann = (float(np.expm1(np.log1p(r).mean() * ppy))
               if len(r) and np.isfinite(ppy) and (r > -1).all() else np.nan)
        lines.append(f"N={n}: 几何年化 {ann:+.2%}（{len(r)} 个信号日）")
    rep.add(SKIP, "Top-N 邻域体检（仅陈列）", "；".join(lines),
            f"当前声明 N={current}。这些几何年化数字仅展示宽度梯度；"
            "本项没有零模型、配对 t 检验或 MDE，不得把某一档读成更优或显著",
            section=SECTION)


def check_entry(wm, close_pm, open_pm, rep):
    ro = _event_returns(wm, open_pm, close_pm, 1)
    rc = _event_returns(wm, close_pm, close_pm, 1)
    z = _paired(ro, rc)
    if not z["n"]:
        rep.skip("入场时点体检", "open/close 可配对样本为空")
        return
    detail = (f"open 入场相对 close 入场 Δmean={z['mean']:+.2%}, t={z['t']:+.2f}, "
              f"n={z['n']}, MDE={z['mde']:.2%}；"
              f"{'有实质差异' if z['passed'] else '不可区分'}。"
              "open 口径避开了打分日收盘到次日开盘的不可成交隔夜段，这是正确的一侧。")
    rep.add(OK, "入场时点体检", detail, "这是执行口径的正向确认，不是参数换档建议", section=SECTION)


def check_signal_consistency(wm, signals, lag, price_days, rep):
    days = list(price_days); pos = {d: i for i, d in enumerate(days)}
    matched = total = 0
    by_date = {d: g for d, g in signals.groupby("date")}
    for exec_d, w in wm.iterrows():
        if exec_d not in pos or pos[exec_d] - lag < 0: continue
        sig_d = days[pos[exec_d] - lag]
        g = by_date.get(sig_d)
        if g is None or g.empty: continue
        mx = g["score"].max(); winners = set(g.loc[g["score"] == mx, "code"].astype(str))
        held = set(map(str, w.index[w > 0]))
        total += 1; matched += int(bool(held & winners))
    if not total:
        rep.skip("信号-持仓自洽", "信号日与执行日无法按 signal_lag 对齐")
        return
    rate = matched / total
    lvl = OK if rate >= .99 else WARN
    rep.add(lvl, "信号-持仓自洽", f"argmax(score) 与实际买入一致 {matched}/{total}（{rate:.1%}）",
            "不一致通常来自排序方向反了、下单面板与信号面板错位，或 tie 处理不同",
            section=SECTION)


def check_deferred_exit(trades, close_pm, flags, rep: AuditReport,
                        wm: pd.DataFrame | None = None,
                        entry_pm: pd.DataFrame | None = None) -> None:
    """跌停封板日按收盘价卖出 ⇒ 现实中卖不掉，出场必须顺延。

    ★ 判板必须用【市场行情价】，不能用【成交价】
    ------------------------------------------
    第一版 e2e 用 `实际卖出价 == down_limit` 判定，得 0/796，据此结论
    「未观察到收盘跌停卖出」。那个判据在构造上不可能成立：成交价含滑点，
    永远不会精确等于跌停价。实测本账 8 笔真跌停里，`实际卖出价` 全部
    ≠ `down_limit`（最极端 002245 市场收盘 10.78、成交价 7.217）。

    正确判据是 `市场 close == down_limit` —— 那 8 笔收盘精确为 −10.0%，
    是真封板。它们合计占全样本总盈亏 −13.41%，不是可忽略的尾巴。

    ★ 本项报事实与顺延后的量级，不建议改策略
    ------------------------------------
    顺延方向不确定（次日可能继续跌也可能回升），所以本项只给
    「这些笔在回测里记了、实盘拿不到，换成次日首个非跌停日是多少」，
    由用户决定怎么处理。
    """
    down_limit = (flags or {}).get("down_limit")
    if down_limit is None:
        rep.skip("跌停出场顺延",
                 "缺 down_limit 列（价格表里加 down_limit/跌停价）；"
                 "补上可回答「有多少笔出场在跌停封板日、顺延后净值差多少」")
        return
    if trades is None or "exit_date" not in getattr(trades, "columns", ()):
        rep.skip("跌停出场顺延", "缺交易明细的 exit_date 列，无法定位出场日")
        return
    events = _trade_events(wm if wm is not None else pd.DataFrame(), trades)
    # ★ 入场价必须用策略真正的入场口径（open 入场就用 open）。
    # 用 close 当入场价会让这里报的毛收益与 §1/§5 那套数字对不上，
    # 同一份报告里出现两个互相矛盾的均值。
    entry_pm = close_pm if entry_pm is None else entry_pm
    days = list(close_pm.index)
    pos = {d: i for i, d in enumerate(days)}
    blocked, deferred, held = [], [], []
    for _, row in events.iterrows():
        e = row.get("exit_date")
        code = row["code"]
        if pd.isna(e) or e not in pos or code not in close_pm.columns:
            continue
        if e not in down_limit.index or code not in down_limit.columns:
            continue
        cl, dl = close_pm.at[e, code], down_limit.at[e, code]
        if not (np.isfinite(cl) and np.isfinite(dl)) or not np.isclose(cl, dl, rtol=1e-6, atol=1e-6):
            continue
        entry = row["entry_date"]
        if (entry not in entry_pm.index or code not in entry_pm.columns
                or not np.isfinite(entry_pm.at[entry, code])):
            continue
        a = float(entry_pm.at[entry, code])
        if a == 0:
            continue
        blocked.append((code, e, float(cl / a - 1.0)))
        # 顺延到出场日之后第一个【非跌停】交易日
        nxt = None
        for j in range(pos[e] + 1, min(pos[e] + 11, len(days))):
            dj = days[j]
            cj = close_pm.at[dj, code] if code in close_pm.columns else np.nan
            dlj = (down_limit.at[dj, code]
                   if dj in down_limit.index and code in down_limit.columns else np.nan)
            if not np.isfinite(cj):
                continue
            if np.isfinite(dlj) and np.isclose(cj, dlj, rtol=1e-6, atol=1e-6):
                continue
            nxt = (dj, float(cj / a - 1.0))
            break
        if nxt is not None:
            deferred.append(nxt[1])
            held.append(float(cl / a - 1.0))
    if not blocked:
        rep.add(OK, "跌停出场顺延",
                "没有任何一笔的出场日收盘价封在跌停板上（判据：市场 close == down_limit）"
                "\n★ 判据用市场行情价，不用成交价 —— 成交价含滑点，永不精确等于跌停价",
                section=SECTION)
        return
    n = len(blocked)
    as_is = float(np.mean([r for _, _, r in blocked]))
    rep.stats["deferred_exit_n"] = n
    rep.stats["deferred_exit_asis_mean"] = as_is
    lines = [f"出场日收盘封在跌停板的交易：{n} 笔（判据：市场 close == down_limit）",
             f"这些笔按回测口径（跌停收盘价成交）平均毛收益 {as_is:+.2%}"]
    if deferred:
        d = np.asarray(deferred) - np.asarray(held)
        rep.stats["deferred_exit_delta_mean"] = float(d.mean())
        lines.append(f"顺延到次个非跌停日后平均毛收益 {float(np.mean(deferred)):+.2%}"
                     f"（{len(deferred)} 笔可顺延，逐笔差 {float(d.mean()):+.2%}）")
    lines.append("明细：" + "、".join(
        f"{c} {pd.Timestamp(e).date()} {r:+.2%}" for c, e, r in blocked[:12]))
    lines.append("★ 判据用市场行情价，不用成交价 —— 成交价含滑点，"
                 "永不精确等于跌停价，用成交价判板会恒得 0 笔")
    rep.add(WARN, "出场撞在跌停板上", "\n".join(lines),
            "跌停封板日按收盘价卖出在现实中卖不掉，这些笔的收益回测记了、实盘拿不到。"
            "顺延方向不确定（次日可能继续跌也可能回升），本项只给量级，"
            "不代替你决定怎么处理", section=SECTION)


def check_vendor_pair(a, b, close_pm, entry_pm, params, rep):
    h = int(params.get("holding_days", 1)); n_top = int(params.get("top_n", 1))
    ra = _rank_returns(a, close_pm, h, n_top, entry_pm)
    rb = _rank_returns(b, close_pm, h, n_top, entry_pm)
    z = _paired(ra, rb)
    if not z["n"]:
        rep.skip("跨供应商配对", "两份信号没有共同可计算的信号日")
        return
    rep.add(OK if not z["passed"] else WARN, "跨供应商配对",
            f"A-B 配对差 {z['mean']:+.2%}（t={z['t']:+.2f}, n={z['n']}, MDE={z['mde']:.2%}）—— "
            f"{'有实质差异' if z['passed'] else '不可区分'}",
            "用于判断 alpha 是策略的还是数据商口径的；结论不包含换供应商建议", section=SECTION)
