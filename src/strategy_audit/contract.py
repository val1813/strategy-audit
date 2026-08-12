"""输入契约：权重面板 + 价格面板。

    weights  date, code, weight        每个调仓日的目标持仓权重
    prices   date, code, close         日频价格（与 factor-audit 同一契约）

★ 为什么要两张表而不是一条净值曲线
--------------------------------
只给净值，能审的只有显著性那一族。换手低估、vw 权重前视、成分变动记账
这三类最值钱的缺陷都需要看到权重本身。而价格面板沿用 factor-audit 的
date/code/close，客户不用为第二个工具重新准备数据。

★ 契约检查只做「能不能算」，不做「算得对不对」
--------------------------------------------
本模块只负责把输入规整成下游能用的形状，并对【会让下游静默算错】的
问题报 BLOCK。业务判断全在各族检查里。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .report import BLOCK, WARN, AuditReport

W_REQUIRED = ("date", "code", "weight")
P_REQUIRED = ("date", "code", "close")

# 权重和偏离 1.0 超过这个量 ⇒ 不是「浮点误差」而是口径问题
GROSS_TOL = 1e-6
# 权重和偏离 1.0 超过这个量 ⇒ BLOCK（下游按 gross=1 归一会改变结论）
GROSS_BLOCK = 0.01


def _norm_dates(d: pd.DataFrame, col: str = "date") -> pd.DataFrame:
    d = d.copy()
    d[col] = pd.to_datetime(d[col])
    return d


def load_weights(w: pd.DataFrame, rep: AuditReport) -> pd.DataFrame | None:
    """规整权重面板。返回 None 表示 BLOCK，下游不该继续。"""
    missing = [c for c in W_REQUIRED if c not in w.columns]
    if missing:
        rep.add(BLOCK, "权重面板必需列缺失", f"缺 {missing}",
                "date/code/weight 是最低要求", section="输入契约")
        return None

    w = _norm_dates(w)
    w["weight"] = pd.to_numeric(w["weight"], errors="coerce")

    n_null = int(w["weight"].isna().sum())
    if n_null:
        # ★ 不能 fillna(0)。权重为 null 有两种截然不同的含义：
        # 「这只票这期不持有」(=0) 与「数据没取到」(=未知)。
        # 当成 0 会把「数据缺失」记成「主动清仓」，而清仓在换手里
        # 是一笔真实交易 —— 凭空造出换手，或凭空抹掉持仓。
        rep.add(BLOCK, "权重有空值",
                f"{n_null:,} 行 weight 为空（共 {len(w):,} 行）",
                "空值有两种含义：不持有(=0) 与 没取到(未知)。"
                "下游 fillna(0) 会把「数据缺失」记成「主动清仓」——"
                "凭空造出一笔换手。请自行明确后再传入",
                section="输入契约")
        return None

    dup = int(w.duplicated(["date", "code"]).sum())
    if dup:
        rep.add(BLOCK, "权重面板有重复行",
                f"{dup:,} 行 (date, code) 重复",
                "重复行会在 groupby 求和时把同一只票的权重加倍",
                section="输入契约")
        return None

    if (w["weight"] < 0).any():
        n_short = int((w["weight"] < 0).sum())
        # 空头不是错误，但本版的换手/成本口径是按多头 gross=1 标定的
        rep.add(WARN, "含空头权重",
                f"{n_short:,} 行 weight < 0",
                "本版换手与成本按多头 gross=1 标定。空头组合的换手分母"
                "（gross vs net）口径不同，下面的成本量级会偏小",
                section="输入契约")

    return w.sort_values(["date", "code"]).reset_index(drop=True)


def load_prices(p: pd.DataFrame, rep: AuditReport) -> pd.DataFrame | None:
    """规整价格面板。"""
    missing = [c for c in P_REQUIRED if c not in p.columns]
    if missing:
        rep.add(BLOCK, "价格面板必需列缺失", f"缺 {missing}",
                "date/code/close 是最低要求（与 factor-audit 同一契约）",
                section="输入契约")
        return None

    p = _norm_dates(p)
    p["close"] = pd.to_numeric(p["close"], errors="coerce")

    dup = int(p.duplicated(["date", "code"]).sum())
    if dup:
        # ★ 实测踩过：daily_clean.parquet 有重复行（2026-05 两日 / 11998 行）。
        # 重复价格会让 pivot 静默取最后一行，而两行价格可能不同。
        rep.add(BLOCK, "价格面板有重复行",
                f"{dup:,} 行 (date, code) 重复",
                "pivot 会静默保留其中一行。实测真实面板出现过此缺陷"
                "（某月两日共 11998 重复行）——先去重并确认保留哪一行",
                section="输入契约")
        return None

    return p.sort_values(["date", "code"]).reset_index(drop=True)


def check_gross(w: pd.DataFrame, rep: AuditReport) -> pd.DataFrame:
    """检查每期权重和，并按 gross=1 归一。

    ★ 归一这件事必须报出来，不能静默做。
    权重和 0.95 有两种可能：留了 5% 现金（真实的择时决策），或者
    权重算错了。前者归一会抹掉一个真实的收益来源（现金拖累），
    后者不归一会让净值口径错。工具不能替客户判断是哪一种。
    """
    g = w.groupby("date")["weight"].sum()
    dev = (g - 1.0).abs()
    worst = float(dev.max()) if len(dev) else 0.0
    rep.stats["gross_max_dev"] = worst

    if worst <= GROSS_TOL:
        rep.add("OK", "权重和", f"{len(g)} 期权重和均为 1（最大偏离 {worst:.2e}）",
                section="输入契约")
        return w

    n_off = int((dev > GROSS_TOL).sum())
    lvl = BLOCK if worst > GROSS_BLOCK else WARN
    rng = f"{float(g.min()):.4f} ~ {float(g.max()):.4f}"
    rep.add(lvl, "权重和不为 1",
            f"{n_off}/{len(g)} 期权重和偏离 1（范围 {rng}，最大偏离 {worst:.4f}）",
            "两种可能，工具无法替你判断：①有意留现金 ⇒ 归一会抹掉现金拖累"
            "这个真实的收益来源；②权重算错 ⇒ 不归一则净值口径错。"
            "下面的换手与收益按 gross=1 归一后计算",
            section="输入契约")
    return w


def to_matrix(w: pd.DataFrame) -> pd.DataFrame:
    """权重长表 → date × code 矩阵，缺失填 0（=不持有）。

    到这一步 null 已在 load_weights 里被 BLOCK 拦掉，所以这里的
    「缺失」只可能是该期确实没有这只票的行 —— 那就是不持有。
    """
    m = w.pivot(index="date", columns="code", values="weight")
    return m.reindex(sorted(m.columns), axis=1).fillna(0.0).sort_index()


def normalize_gross(m: pd.DataFrame) -> pd.DataFrame:
    """按每期 gross（绝对值和）归一到 1。"""
    gross = m.abs().sum(axis=1).replace(0.0, np.nan)
    return m.div(gross, axis=0).fillna(0.0)
