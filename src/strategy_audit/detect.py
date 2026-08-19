"""自动识别用户上传的表：他给什么，我们就认出来是什么。

★ 为什么必须有这一层
------------------
第一版要求 `date/code/weight` + `date/code/close` 两张精确命名的表，
缺列就 BLOCK。真实用户手上的东西五花八门：列叫 `trade_date`/`日期`、
持仓表是宽表（行=日期、列=股票）、只有一条净值曲线、收益率是百分数
形式（1.5 表示 1.5%）……

要求他先改成我们的格式，等于把工具的门槛变成他的工作量 —— 而他
本来就是因为不确定自己的回测有没有问题才来的。

★ 本模块只做识别，不做判断
------------------------
识别结果【必须报给用户看】，因为猜错比不猜更糟：把价格当净值会让
所有检查算在错的东西上，而报告照样打印得像模像样。所以每个识别
都带 confidence 和依据，报告里原样列出，让用户能一眼纠正。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# ---------------- 列名别名 ----------------
# 只放【实际见过】的写法，不做无边界的猜测。
# 中文列名必须支持：国内用户的导出表大量使用中文表头。
ALIASES = {
    "date": ("date", "trade_date", "tradedate", "datetime", "dt", "day", "time",
             "时间", "日期", "交易日", "交易日期", "调仓日", "调仓日期"),
    "code": ("code", "symbol", "ticker", "sid", "secid", "sec_id", "instrument",
             "stock", "stock_code", "asset", "security", "permno", "wind_code", "ts_code",
             "代码", "股票", "股票代码", "证券代码", "标的", "合约"),
    "weight": ("weight", "w", "wgt", "target_weight", "target_w", "pos",
               "holding", "alloc", "allocation",
               "权重", "目标权重", "持仓", "仓位", "配置"),
    "close": ("close", "close_price", "closeprice", "px", "price", "prc",
              "adj_close", "adjclose", "close_adj", "vwap", "settle",
              "收盘", "收盘价", "价格", "复权价", "后复权", "前复权"),
    "nav": ("nav", "net_value", "netvalue", "equity", "cum", "cumnav",
            "cum_nav", "cumulative", "curve", "value", "balance",
            "净值", "累计净值", "单位净值", "权益", "资金曲线", "累计收益"),
    "ret": ("ret", "return", "returns", "rtn", "r", "pnl", "profit",
            "period_return", "daily_return", "chg", "pct_chg", "pct_change",
            "收益", "收益率", "回报", "涨跌幅", "日收益", "期间收益"),
    # 成交额（族五容量用）。★ 与成交量(vol/股数)区分：这里要的是【金额】。
    # 名字带 vol 的列在真实数据里既可能是成交量也可能是波动率，太危险，不收。
    "amount": ("amount", "amt", "turnover_value", "dollar_volume", "dollarvol",
               "value_traded", "traded_value", "money", "notional",
               "成交额", "成交金额", "交易额", "金额"),
    "cap": ("cap", "mktcap", "market_cap", "marketcap", "mv", "size",
            "float_cap", "circ_mv", "total_mv",
            "市值", "总市值", "流通市值"),
    "bench": ("bench", "benchmark", "bmk", "index", "index_ret", "bench_ret",
              "基准", "基准收益", "指数", "指数收益"),
    "net": ("net", "net_ret", "net_return", "after_cost", "ret_net",
            "净收益", "扣费后", "税后"),
    "gross": ("gross", "gross_ret", "gross_return", "before_cost", "ret_gross",
              "毛收益", "扣费前"),
}

# 每期权重和落在这个范围内 ⇒ 像是权重（而不是价格/市值）
GROSS_LO, GROSS_HI = 0.5, 2.0


def _norm(s: str) -> str:
    return str(s).strip().lower().replace(" ", "_").replace("-", "_")


def match_column(cols, kind: str) -> str | None:
    """在列名里找某个语义的列。完全匹配优先，其次子串。"""
    al = ALIASES[kind]
    norm = {c: _norm(c) for c in cols}
    for c, n in norm.items():
        if n in al:
            return c
    # 子串匹配：`close_adj_hfq` 这类
    for c, n in norm.items():
        for a in al:
            if len(a) >= 3 and a in n:
                return c
    return None


@dataclass
class Detected:
    """一张表的识别结果。"""
    kind: str                     # weights | prices | series | unknown
    frame: pd.DataFrame | None = None
    series: pd.Series | None = None
    role: str = ""                # series 的角色：nav/ret/bench/net
    layout: str = ""              # long | wide | single
    columns: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    source: str = ""


def _is_datelike(s: pd.Series) -> bool:
    """这一列能不能当日期用。"""
    if pd.api.types.is_datetime64_any_dtype(s):
        return True
    sample = s.dropna().head(50)
    if sample.empty:
        return False
    try:
        # 纯整数列（如 20210104）也算
        if pd.api.types.is_integer_dtype(sample):
            v = sample.astype(str)
            return bool(v.str.len().isin((8,)).all())
        pd.to_datetime(sample, errors="raise")
        return True
    except Exception:
        return False


def _to_datetime(s: pd.Series) -> pd.Series:
    if pd.api.types.is_integer_dtype(s):
        return pd.to_datetime(s.astype(str), format="%Y%m%d", errors="coerce")
    return pd.to_datetime(s, errors="coerce")


def classify_series(s: pd.Series) -> tuple[str, list[str]]:
    """判断一条数值序列是净值还是收益率。

    ★ 这个判别必须保守并把依据说出来 —— 猜错方向会让整份报告算错。
    依据（按可靠性排序）：
      1. 有负值 ⇒ 一定不是净值（净值不会为负）
      2. 全为正且均值远大于 1 ⇒ 净值（如 1.0 → 2.3）
      3. 绝对值都很小（<0.5）⇒ 收益率
      4. 单调性：净值通常不会每期正负交替
    """
    v = pd.Series(s).dropna().astype(float)
    notes: list[str] = []
    if len(v) < 3:
        return "unknown", ["序列过短（<3 期），无法判别净值/收益率"]

    has_neg = bool((v < 0).any())
    med_abs = float(v.abs().median())
    mean_v = float(v.mean())
    sign_flips = float((np.sign(v.values[1:]) != np.sign(v.values[:-1])).mean())

    if has_neg:
        notes.append(f"含负值（最小 {v.min():.4f}）⇒ 不是净值")
        return "ret", notes
    if med_abs > 0.5:
        notes.append(f"全为正且中位数 {med_abs:.3f} > 0.5 ⇒ 像净值/价格")
        if mean_v > 0.5:
            return "nav", notes
    if med_abs < 0.2:
        notes.append(f"中位数绝对值 {med_abs:.4f} < 0.2 ⇒ 像收益率")
        if sign_flips > 0.1:
            notes.append(f"{sign_flips:.0%} 的相邻期符号翻转 ⇒ 收益率")
        return "ret", notes
    notes.append(f"中位数 {med_abs:.3f} 落在模糊区间（0.2~0.5）")
    return "unknown", notes


def looks_like_percent(s: pd.Series) -> bool:
    """收益率是不是写成了百分数（1.5 表示 1.5%）。

    ★ 实测最常见的静默口径错。日频收益率的标准差约 0.02；
    若写成百分数则约 2.0 —— 差 100 倍，会让所有年化数字荒谬。
    """
    v = pd.Series(s).dropna().astype(float)
    if len(v) < 5:
        return False
    return float(v.abs().median()) > 0.15 and float(v.abs().max()) < 100


def detect_frame(df: pd.DataFrame, source: str = "") -> Detected:
    """识别一张表是什么。"""
    if df is None or len(df) == 0:
        return Detected("unknown", source=source,
                        notes=["表是空的"])

    cols = list(df.columns)
    c_date = match_column(cols, "date")

    # ---- 情形 A：索引就是日期（宽表或单列序列）----
    if c_date is None and _is_datelike(pd.Series(df.index)):
        df = df.copy()
        df.insert(0, "__date__", _to_datetime(pd.Series(df.index)).values)
        c_date = "__date__"
        cols = list(df.columns)

    if c_date is None:
        return Detected("unknown", source=source,
                        notes=[f"找不到日期列（列名：{cols[:8]}）"])

    d = df.copy()
    d[c_date] = _to_datetime(d[c_date])
    n_bad = int(d[c_date].isna().sum())
    notes = []
    if n_bad:
        notes.append(f"{n_bad} 行日期无法解析，已丢弃")
        d = d[d[c_date].notna()]

    c_code = match_column(cols, "code")
    num_cols = [c for c in d.columns
                if c != c_date and pd.api.types.is_numeric_dtype(d[c])]

    # ---- 情形 B：长表（有 code 列）----
    if c_code is not None:
        c_w = match_column(cols, "weight")
        c_p = match_column(cols, "close")
        c_cap = match_column(cols, "cap")

        # 平台持仓快照常把【股数/市值】叫 position。即使其数值恰好每期
        # 加总为 1，也不能静默猜成目标权重；须由导出端明确命名 target_weight。
        if c_w is not None and _norm(c_w) in ("position", "positions"):
            return Detected("unknown", layout="long",
                            notes=[f"列 `{c_w}` 像持仓数量/市值，不是可确认的目标权重；"
                                   "请导出 target_weight/weight"], source=source)

        out = d.rename(columns={c_date: "date", c_code: "code"})
        colmap = {"date": c_date, "code": c_code}

        if c_w is not None:
            out = out.rename(columns={c_w: "weight"})
            colmap["weight"] = c_w
            keep = ["date", "code", "weight"]
            if c_p is not None:
                out = out.rename(columns={c_p: "close"})
                colmap["close"] = c_p
                keep.append("close")
            notes.append(f"长表，识别为【权重】：{c_w} → weight")
            return Detected("weights", frame=out[keep], layout="long",
                            columns=colmap, notes=notes, source=source)

        if c_p is not None:
            out = out.rename(columns={c_p: "close"})
            colmap["close"] = c_p
            keep = ["date", "code", "close"]
            if c_cap is not None:
                out = out.rename(columns={c_cap: "cap"})
                colmap["cap"] = c_cap
                keep.append("cap")
            # 成交额/成交量：族五（容量与可成交性）要用，有就带上，没有就跳过该族
            c_amt = match_column(cols, "amount")
            if c_amt is not None:
                out = out.rename(columns={c_amt: "amount"})
                colmap["amount"] = c_amt
                keep.append("amount")
                notes.append(f"额外带上成交额：{c_amt} → amount（可审容量）")
            notes.append(f"长表，识别为【价格】：{c_p} → close")
            return Detected("prices", frame=out[keep], layout="long",
                            columns=colmap, notes=notes, source=source)

        # 有 code 但列名不认识 ⇒ 用数值特征猜
        if len(num_cols) == 1:
            v = num_cols[0]
            g = d.groupby(d[c_date])[v].sum()
            in_range = float(((g > GROSS_LO) & (g < GROSS_HI)).mean())
            if in_range > 0.8:
                out = out.rename(columns={v: "weight"})
                colmap["weight"] = v
                notes.append(
                    f"长表，列 `{v}` 每期求和 {in_range:.0%} 落在 "
                    f"{GROSS_LO}~{GROSS_HI} ⇒ 判为【权重】")
                return Detected("weights", frame=out[["date", "code", "weight"]],
                                layout="long", columns=colmap, notes=notes,
                                source=source)
            out = out.rename(columns={v: "close"})
            colmap["close"] = v
            notes.append(f"长表，列 `{v}` 每期求和不像权重 ⇒ 判为【价格】")
            return Detected("prices", frame=out[["date", "code", "close"]],
                            layout="long", columns=colmap, notes=notes,
                            source=source)

        notes.append(f"有 code 列但认不出数值列语义（数值列：{num_cols[:6]}）")
        return Detected("unknown", layout="long", notes=notes, source=source)

    # ---- 情形 C：单列序列 ----
    if len(num_cols) == 1:
        v = num_cols[0]
        s = pd.Series(d[v].astype(float).values, index=d[c_date]).sort_index()
        role = None
        for kind in ("nav", "net", "bench", "ret"):
            if match_column([v], kind):
                role = kind
                break
        guessed, why = classify_series(s)
        if role is None:
            role = guessed
            notes.append(f"列 `{v}` 名字认不出，按数值判为 {role}：" + "；".join(why))
        else:
            notes.append(f"列 `{v}` 按名字识别为 {role}")
            if role in ("nav", "ret") and guessed != "unknown" and guessed != role:
                notes.append(
                    f"★ 但数值特征更像 {guessed}（{'；'.join(why)}）—— 请确认")
        if role == "ret" and looks_like_percent(s):
            notes.append(
                f"★ 收益率绝对值中位数 {float(s.abs().median()):.3f} 偏大，"
                "疑似写成了【百分数】（1.5 表示 1.5%）。"
                "若是，请先除以 100，否则所有年化数字会差 100 倍")
        return Detected("series", series=s, role=role, layout="single",
                        columns={"date": c_date, "value": v},
                        notes=notes, source=source)

    # ---- 情形 D：宽表（行=日期，多列=标的）----
    if len(num_cols) >= 2:
        wide = d.set_index(c_date)[num_cols].sort_index()
        row_sum = wide.abs().sum(axis=1)
        in_range = float(((row_sum > GROSS_LO) & (row_sum < GROSS_HI)).mean())
        long = (wide.stack().rename("value").reset_index())
        long.columns = ["date", "code", "value"]
        if in_range > 0.8:
            long = long.rename(columns={"value": "weight"})
            long = long[long["weight"] != 0.0]
            notes.append(f"宽表（{len(num_cols)} 列），每行绝对值求和 "
                         f"{in_range:.0%} 落在 {GROSS_LO}~{GROSS_HI} ⇒ 判为【权重】")
            return Detected("weights", frame=long, layout="wide",
                            columns={"date": c_date}, notes=notes, source=source)
        long = long.rename(columns={"value": "close"}).dropna(subset=["close"])
        notes.append(f"宽表（{len(num_cols)} 列），每行求和不像权重 ⇒ 判为【价格】")
        return Detected("prices", frame=long, layout="wide",
                        columns={"date": c_date}, notes=notes, source=source)

    notes.append("表里没有可用的数值列")
    return Detected("unknown", notes=notes, source=source)


def detect_all(objs: list) -> list[Detected]:
    """识别一组输入。每个元素可以是 DataFrame / Series / (name, obj)。"""
    out = []
    for i, o in enumerate(objs):
        name = ""
        if isinstance(o, tuple) and len(o) == 2:
            name, o = o
        src = name or f"输入{i + 1}"
        if isinstance(o, pd.Series):
            s = o.dropna().astype(float)
            role, why = classify_series(s)
            notes = [f"Series，按数值判为 {role}：" + "；".join(why)]
            if role == "ret" and looks_like_percent(s):
                notes.append("★ 疑似百分数形式，请确认是否需要除以 100")
            out.append(Detected("series", series=s, role=role, layout="single",
                                notes=notes, source=src))
        elif isinstance(o, pd.DataFrame):
            out.append(detect_frame(o, source=src))
        else:
            out.append(Detected("unknown", source=src,
                                notes=[f"不支持的类型 {type(o).__name__}"]))
    return out
