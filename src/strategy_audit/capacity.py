"""族五：容量与可成交性 —— 这些收益你到底拿不拿得到。

审的是「按这份持仓,能管多少钱、有多少笔根本下不去」,四项:

    ① 资金容量上限   按「单日不超过成交额 X%」反解可管理规模
    ② 不可成交权重   调仓日涨跌停/停牌的标的占多少权重
    ③ 流动性集中度   收益是不是来自最难成交的那批标的
    ④ 规模衰减曲线   规模放大时冲击成本吃掉多少超额

★ 这一族和族一（换手与成本）的分界
--------------------------------
族一问「你扣的成本对不对」—— 那是**记账**问题,答案与你管多少钱无关。
族五问「这份持仓能承载多少钱」—— 那是**容量**问题,答案是一个金额。
同一个策略可以成本记得完全正确,同时只能管 2000 万。

★ 这一族为什么不出 BLOCK
----------------------
BLOCK 的语义是「净值不可信,先修」。容量不足不会让回测的净值算错 ——
它让这份净值**与你的实际规模无关**。一个只能管 2000 万的策略,
回测净值本身没有错,错的是拿它去说明一个 20 亿的产品。
所以本族最高只到 WARN。唯一例外见 ② 的说明。

★ 不可成交检测是【推断】,不是事实
------------------------------
工具只有 close,没有涨跌停标志位、没有停牌标志位。所以:
  · 涨跌停按「|日收益| ≥ 板块限幅 − 容差」推断,会有误判
    （新股上市前 5 日无限幅、ST 是 5%、北交所 30%）
  · 停牌按「该日无价格」推断,而那也可能只是数据缺失
两者都会**高估**不可成交比例（把正常涨跌当成涨停）。方向是保守的,
但仍是推断 —— 报告里必须写明这是推断且给出板块判定依据。

当前公开输入契约只保留 close/amount；即使平台导出真的 `limit_up` /
`suspended` 标志位，它们也尚未接入本检查。请不要把推断结果当作交易状态事实。
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .core import (_seg_gross, align, drift_weights, period_returns,
                   periods_per_year, turnover)
from .report import OK, WARN, AuditReport

SECTION = "容量与可成交性"

# 单日下单不超过该标的日成交额的这个比例（用于反解容量）。
# ★ 10% 是业界常用的保守上限；再高会显著推动价格,而我们没有冲击模型
# 能可信地外推。这个数会在报告里明说,因为容量结论对它线性敏感。
ADV_PARTICIPATION = 0.10

# 涨跌停判定：按代码前缀分板的日内限幅
LIMIT_BY_BOARD = {
    "sh.60": 0.10,   # 上海主板
    "sz.00": 0.10,   # 深圳主板
    "sz.30": 0.20,   # 创业板
    "sh.68": 0.20,   # 科创板
    "bj.": 0.30,     # 北交所
}
DEFAULT_LIMIT = 0.10

# 判定容差：实际收益到限幅的距离小于这个数就算触板
# （复权价算出的收益与原始涨跌幅有微小差异）
LIMIT_TOL = 0.002

# ST / *ST 股的日内限幅（主板 ST 为 5%）。
# ★ 工具无从得知一只票在某一天是不是 ST —— 那要看当时的公告。
# 没有标志位时按板块限幅判，会把 ST 股的 ±5% 触板当成没触板，
# 方向是【低估】不可成交。有 st 标志位面板就传进来。
ST_LIMIT = 0.05

# 不可成交权重超过这些比例 ⇒ 报警
UNTRADABLE_WARN = 0.05
UNTRADABLE_HIGH = 0.15

# 容量低于这个金额（元）⇒ 值得提醒：多数机构产品做不了
CAPACITY_WARN_CNY = 5e8

# 规模衰减曲线要测的规模档（元）
SIZE_GRID = (1e7, 5e7, 1e8, 5e8, 1e9, 5e9)

# 冲击成本系数：cost_bp ≈ IMPACT_K * sqrt(参与率)
# ★ 平方根律是行业惯例（Almgren 等）,系数量级取决于市场。
# 这个数不可能对所有标的都准,所以报告给的是【衰减形状】而不是精确成本。
IMPACT_K = 100.0


_DIGITS = re.compile(r"\d{6}")

# 按【6 位数字代码】判板 —— 这才是板块的真实标识，交易所前缀只是包装。
# 键是数字段的前缀，按长度从长到短匹配（688 要先于 68 之类的歧义不存在，
# 但 3 位前缀必须先于 2 位，所以下面显式按长度排序）。
LIMIT_BY_DIGITS = {
    "688": 0.20,   # 科创板
    "689": 0.20,   # 科创板 CDR
    "300": 0.20,   # 创业板
    "301": 0.20,   # 创业板
    "302": 0.20,   # 创业板
    "430": 0.30,   # 北交所
    "830": 0.30,   # 北交所
    "831": 0.30,   # 北交所
    "832": 0.30,   # 北交所
    "833": 0.30,   # 北交所
    "834": 0.30,   # 北交所
    "835": 0.30,   # 北交所
    "836": 0.30,   # 北交所
    "837": 0.30,   # 北交所
    "838": 0.30,   # 北交所
    "839": 0.30,   # 北交所
    "870": 0.30,   # 北交所
    "871": 0.30,   # 北交所
    "872": 0.30,   # 北交所
    "873": 0.30,   # 北交所
    "920": 0.30,   # 北交所（2024 起新号段）
}


def _board_limit(code: str) -> float:
    """标的的日内限幅。

    ★ 必须按【数字段】判板，不能按交易所前缀判。
    实测：只认 `sh.`/`sz.` 前缀时，这些同样常见的写法全部落到默认 10% ——

        300750.SZ  创业板 20% ⇒ 误判 10%
        688981.SH  科创板 20% ⇒ 误判 10%
        430047.BJ  北交所 30% ⇒ 误判 10%
        300750     裸 6 位     ⇒ 误判 10%

    限幅判小了会把正常涨跌当成触板（|r| ≥ 10% 就报涨停），于是
    创业板/科创板的不可成交比例被系统性【高估】。方向虽保守，
    但一个 20% 限幅的标的涨 12% 被报成「涨停下不去」是错的。
    """
    s = str(code).upper()
    m = _DIGITS.search(s)
    if m:
        d = m.group(0)
        for pre in sorted(LIMIT_BY_DIGITS, key=len, reverse=True):
            if d.startswith(pre):
                return LIMIT_BY_DIGITS[pre]
        return DEFAULT_LIMIT
    # 没有 6 位数字段时退回前缀匹配（保留对旧写法的兼容）
    for pre, lim in LIMIT_BY_BOARD.items():
        if s.lower().startswith(pre):
            return lim
    return DEFAULT_LIMIT


def board_limits(codes) -> pd.Series:
    """每只标的的日内限幅。"""
    return pd.Series({c: _board_limit(c) for c in codes})


def amount_matrix(prices: pd.DataFrame) -> pd.DataFrame | None:
    """价格长表 → date × code 成交额矩阵。没有 amount 列返回 None。"""
    if prices is None or "amount" not in prices.columns:
        return None
    m = prices.pivot(index="date", columns="code", values="amount")
    return m.reindex(sorted(m.columns), axis=1).sort_index()


# ---------------- ① 资金容量上限 ----------------

def _prev_weights(wm: pd.DataFrame, pm: pd.DataFrame | None,
                  prev, cur) -> pd.Series:
    """上期权重在本期调仓【时点】的实际形态。

    给了价格就用漂移后的权重（正确口径）；没有价格才退回上期目标权重。
    ★ 这个函数存在的唯一目的是防止两处 Δw 各写一遍又写歪。
    """
    w0 = wm.loc[prev]
    if pm is None:
        return w0
    g = _seg_gross(pm.reindex(columns=wm.columns), prev, cur)
    if not np.isfinite(g).any():
        return w0
    return drift_weights(w0, g)


def capacity_cny(wm: pd.DataFrame, am: pd.DataFrame,
                 participation: float = ADV_PARTICIPATION,
                 to: pd.DataFrame | None = None,
                 pm: pd.DataFrame | None = None) -> pd.DataFrame:
    """逐调仓日的资金容量上限（元）。

    对每只要交易的标的:  可下单金额 = participation × 该日成交额
    该标的对应的组合规模 = 可下单金额 / |Δw_i|
    整个组合的容量 = 所有要交易标的里最紧的那个（min）。

    ★ 用 Δw 而不是 w：容量由【要交易的量】决定,不是持仓量。
    一只权重 5% 但从不调整的标的不构成容量约束。

    ★ Δw 必须按【漂移调整】口径算,不能用目标-目标之差
    ------------------------------------------------
    给了 pm 时用 w_t − w̃_{t−1}（w̃ = 上期权重被价格推着漂移后归一）。
    这与 core.turnover 的 drift_adj 口径一致,也是族一反复强调的正确口径。

    实测（补测试时抓到的）：目标权重恒定的组合按目标-目标之差算出
    Δw ≡ 0,于是容量与不可成交【每一期都被跳过】,报告只说
    「没有可用数据」—— 而同一份持仓的漂移调整换手是 3.09%/期。
    价格涨了就得卖掉一部分再买回来,那笔交易是真的要下单的。
    最需要审的情形（买入持有型组合）恰好被静默跳过了。
    """
    wm2, am2 = wm.align(am, join="left", axis=1)
    am2 = am2.reindex(index=wm2.index).ffill()
    dates = list(wm2.index)
    rows = []
    for prev, cur in zip(dates[:-1], dates[1:]):
        dw = (wm2.loc[cur] - _prev_weights(wm2, pm, prev, cur)).abs()
        traded = dw[dw > 1e-12]
        if traded.empty:
            continue
        adv = am2.loc[cur, traded.index]
        ok = np.isfinite(adv) & (adv > 0)
        if not ok.any():
            continue
        cap_i = (participation * adv[ok]) / traded[ok]
        rows.append(dict(date=cur,
                         capacity=float(cap_i.min()),
                         binding=str(cap_i.idxmin()),
                         n_traded=int(ok.sum()),
                         median_capacity=float(cap_i.median())))
    return pd.DataFrame(rows).set_index("date") if rows else pd.DataFrame()


def _fmt_cny(x: float) -> str:
    if not np.isfinite(x):
        return "n/a"
    if x >= 1e8:
        return f"{x / 1e8:.2f} 亿"
    if x >= 1e4:
        return f"{x / 1e4:.0f} 万"
    return f"{x:.0f} 元"


def check_capacity(wm: pd.DataFrame, am: pd.DataFrame,
                   rep: AuditReport,
                   pm: pd.DataFrame | None = None) -> pd.DataFrame:
    """① 资金容量上限。"""
    if am is None:
        rep.skip("资金容量上限", "价格面板没有成交额列（amount/成交额）")
        return pd.DataFrame()
    d = capacity_cny(wm, am, pm=pm)
    if not len(d):
        rep.skip("资金容量上限", "没有可用的成交额与换手数据")
        return d

    med = float(d["capacity"].median())
    p10 = float(d["capacity"].quantile(0.10))
    rep.stats["capacity_cny_median"] = med
    rep.stats["capacity_cny_p10"] = p10

    # 最常成为瓶颈的标的
    top = d["binding"].value_counts()
    who = "、".join(f"{k}({v}次)" for k, v in top.head(3).items())

    detail = (f"按「单日下单不超过该标的成交额 {ADV_PARTICIPATION:.0%}」,"
              f"可管理规模中位数 {_fmt_cny(med)}"
              f"（最紧的 10% 调仓日只有 {_fmt_cny(p10)}）"
              f"\n最常成为瓶颈的标的：{who}"
              f"\n★ 容量对参与率假设线性敏感：参与率放宽到 20% 容量翻倍")
    if med < CAPACITY_WARN_CNY:
        rep.add(WARN, "资金容量偏小", detail,
                f"这份持仓的容量中位数 {_fmt_cny(med)}。"
                "回测净值本身没有错,但拿它说明一个更大规模的产品就是错的 —— "
                "超出容量后冲击成本会吃掉超额（见规模衰减曲线）",
                section=SECTION)
    else:
        rep.add(OK, "资金容量上限", detail, section=SECTION)
    return d


# ---------------- ② 不可成交权重 ----------------

def untradable_weight(wm: pd.DataFrame, pm: pd.DataFrame,
                      rm: pd.DataFrame | None = None,
                      st: pd.DataFrame | None = None,
                      flags: dict | None = None) -> pd.DataFrame:
    """逐调仓日:要建/调的仓里有多少权重当天根本下不去。

    涨跌停按「日收益 ≥ +限幅」/「≤ −限幅」推断；停牌按「该日无价格」推断。

    ★ 必须【区分方向】—— 涨停和跌停挡住的是相反的操作
    ------------------------------------------------
        涨停（封在涨停板）  买不进，但【卖得出】
        跌停（封在跌停板）  卖不出，但【买得进】

    第一版用 `|r| ≥ 限幅` 一律算不可成交，于是「涨停日要卖出」和
    「跌停日要买入」这两类【能正常成交】的交易被误判。盲测实测
    77 个触板标的-期里有 69 个是这两类（涨停要卖 56、跌停要买 13），
    不可成交权重被高估 160%（1.30% vs 真实 0.50%）。

    方向仍然是保守的（高估不可成交），但高估 2.6 倍就不再是「保守」，
    而是把一份能执行的策略报成执行不了。

    st  可选的 ST 标志位面板（date × code 的 bool）。A 股 ST 股限幅 5%，
        没有这个标志位时按板块限幅判 —— 会把 ST 股的正常涨跌（±5% 触板）
        当成没触板，方向是【低估】不可成交。有就传进来。
    """
    wm2, pm2 = align(wm, pm)
    if rm is None:
        # ★ 必须显式 fill_method=None。pandas 的默认值是 'pad'（已弃用），
        # 那会先把缺价日【前向填充】再算收益 —— 停牌日于是得到 ret=0
        # 而不是 NaN，复牌那天的跳空也被摊掉。本函数正是要靠
        # 「该日无价格」推断停牌，填充等于把要找的信号先擦掉。
        rm = pm2.pct_change(fill_method=None)
    lim = board_limits(wm2.columns)
    dates = list(wm2.index)
    rows = []
    for prev, cur in zip(dates[:-1], dates[1:]):
        if cur not in rm.index:
            continue
        # ★ 同 capacity_cny：Δw 必须按漂移调整口径算。
        # 目标权重恒定时目标-目标之差恒为 0，于是每期都被跳过 ——
        # 而价格漂移逼出来的交易是真的要下单、真的会撞上涨跌停的。
        dw_signed = wm2.loc[cur] - _prev_weights(wm2, pm2, prev, cur)
        dw = dw_signed.abs()
        mask = dw > 1e-12
        traded = dw[mask]
        if traded.empty:
            continue
        idx = traded.index
        r = rm.loc[cur, idx]
        L = lim.reindex(idx)
        # ST 股限幅 5%（有标志位时按它覆盖板块限幅）
        if st is not None and cur in st.index:
            is_st = st.reindex(index=[cur], columns=idx).iloc[0].fillna(False)
            L = L.where(~is_st.astype(bool), ST_LIMIT)
        up = (r >= (L - LIMIT_TOL)).fillna(False)      # 涨停：买不进
        down = (r <= -(L - LIMIT_TOL)).fillna(False)   # 跌停：卖不出
        flags = flags or {}
        up_limit = flags.get("up_limit")
        down_limit = flags.get("down_limit")
        suspended = flags.get("is_suspended")
        if up_limit is not None and cur in up_limit.index:
            ul = up_limit.loc[cur].reindex(idx)
            up = (pm2.loc[cur, idx] >= ul * (1 - LIMIT_TOL)).fillna(False)
        if down_limit is not None and cur in down_limit.index:
            dl = down_limit.loc[cur].reindex(idx)
            down = (pm2.loc[cur, idx] <= dl * (1 + LIMIT_TOL)).fillna(False)
        buying = dw_signed[mask] > 0
        no_price = ~np.isfinite(pm2.loc[cur, idx])
        if suspended is not None and cur in suspended.index:
            no_price = (no_price | suspended.loc[cur].reindex(idx).fillna(False).astype(bool))
        # ★ 方向感知：涨停只挡买入、跌停只挡卖出、无价格两边都挡
        blocked = ((up & buying) | (down & ~buying) | no_price)
        rows.append(dict(date=cur,
                         w_traded=float(traded.sum()),
                         w_blocked=float(traded[blocked].sum()),
                         n_blocked=int(blocked.sum()),
                         n_limit=int(((up & buying) | (down & ~buying)).sum()),
                         n_nopx=int(no_price.sum()),
                         # 诊断用：被方向判定【放行】的触板笔数
                         n_limit_ok=int(((up & ~buying) | (down & buying)).sum())))
    return pd.DataFrame(rows).set_index("date") if rows else pd.DataFrame()


def check_untradable(wm: pd.DataFrame, pm: pd.DataFrame,
                     rep: AuditReport, *, st: pd.DataFrame | None = None,
                     flags: dict | None = None) -> pd.DataFrame:
    """② 调仓日不可成交的权重占比。"""
    d = untradable_weight(wm, pm, st=st, flags=flags)
    if not len(d):
        rep.skip("不可成交权重", "没有可用的调仓与价格数据")
        return d

    share = (d["w_blocked"] / d["w_traded"].replace(0, np.nan)).dropna()
    if share.empty:
        rep.skip("不可成交权重", "没有需要交易的权重")
        return d
    med = float(share.median())
    worst = float(share.max())
    n_lim = int(d["n_limit"].sum())
    n_npx = int(d["n_nopx"].sum())
    rep.stats["untradable_share_median"] = med
    rep.stats["untradable_share_max"] = worst

    n_ok = int(d["n_limit_ok"].sum()) if "n_limit_ok" in d else 0
    rep.stats["untradable_n_limit"] = n_lim
    rep.stats["untradable_n_limit_passed"] = n_ok
    source_note = ("【事实：用户提供标志位】" if flags or st is not None else
                   "【推断：按代码猜板，会误判】")
    detail = (f"调仓日需要交易的权重里,不可成交的占比中位数 {med:.1%}"
              f"（最差一期 {worst:.1%}）"
              f"\n其中真正挡住的涨跌停 {n_lim} 个标的-期"
              f"（涨停要买 / 跌停要卖）、无价格（疑似停牌）{n_npx} 个"
              f"\n另有 {n_ok} 个标的-期触板但【方向不挡】"
              f"（涨停要卖、跌停要买都能成交）—— 已放行,不计入上面的比例"
              f"\n★ 数据来源：{source_note}。按 6 位数字代码判板"
              f"（主板 10%、创业板/科创板 20%、北交所 30%）。"
              f"新股前 5 日无限幅、ST 是 5%（可传 st 标志位面板覆盖）"
              f"—— 仍会误判")
    if med > UNTRADABLE_HIGH:
        rep.add(WARN, "大量权重当天下不去", detail,
                f"中位 {med:.1%} 的待交易权重推断为不可成交。"
                "回测假设按收盘价全额成交 ⇒ 这部分收益是拿不到的。"
                "建议:①用真实的涨跌停/停牌标志位重跑 "
                "②把不可成交的仓位顺延到下一日并重算净值",
                section=SECTION)
    elif med > UNTRADABLE_WARN:
        rep.add(WARN, "部分权重当天下不去", detail,
                f"中位 {med:.1%} 待交易权重不可成交,量级不大但真实存在。"
                "若策略信号与涨跌停相关（如动量、涨停板战法）,"
                "这个比例会系统性偏高,不是随机损耗",
                section=SECTION)
    else:
        rep.add(OK, "不可成交权重", detail, section=SECTION)
    return d


# ---------------- ③ 流动性集中度 ----------------

def check_liquidity_tilt(wm: pd.DataFrame, am: pd.DataFrame, pm: pd.DataFrame,
                         rep: AuditReport) -> None:
    """③ 收益是不是来自最难成交的那批标的。

    ★ 这一项回答的是「容量约束会不会连带杀掉 alpha」。
    若超额集中在流动性最差的分位,那么放大规模时**先被迫放弃的
    恰好是赚钱的那批** —— 容量损失不是线性的。
    """
    if am is None:
        rep.skip("流动性集中度", "价格面板没有成交额列")
        return
    wm2, pm2 = align(wm, pm)
    am2 = am.reindex(index=wm2.index, columns=wm2.columns).ffill()
    dates = list(wm2.index)
    lo_r, hi_r, lo_w, hi_w = [], [], [], []
    for prev, cur in zip(dates[:-1], dates[1:]):
        w = wm2.loc[prev]
        held = w[w != 0.0].index
        if len(held) < 6:
            continue
        adv = am2.loc[prev, held]
        if cur not in pm2.index or prev not in pm2.index:
            continue
        r = (pm2.loc[cur, held] / pm2.loc[prev, held] - 1.0)
        ok = np.isfinite(adv) & np.isfinite(r) & (adv > 0)
        if int(ok.sum()) < 6:
            continue
        # ★ w 是全列索引、ok 只覆盖 held ⇒ 必须先把 w 收窄到 held 再筛，
        # 否则 pandas 报 Unalignable boolean Series（布尔索引器与被索引对象
        # 索引不一致）。这类错在合成小面板上未必触发，真面板一跑就炸。
        a, rr, ww = adv[ok], r[ok], w.reindex(held)[ok]
        q = a.rank(pct=True)
        lo = q <= 0.33
        hi = q >= 0.67
        if lo.sum() < 2 or hi.sum() < 2:
            continue
        # 组内按权重加权的收益贡献
        lo_r.append(float((ww[lo] * rr[lo]).sum() / max(ww[lo].sum(), 1e-12)))
        hi_r.append(float((ww[hi] * rr[hi]).sum() / max(ww[hi].sum(), 1e-12)))
        lo_w.append(float(ww[lo].sum()))
        hi_w.append(float(ww[hi].sum()))

    if len(lo_r) < 6:
        rep.skip("流动性集中度", f"有效期数只有 {len(lo_r)}（需要 ≥6）")
        return

    lo_m, hi_m = float(np.mean(lo_r)), float(np.mean(hi_r))
    diff = lo_m - hi_m
    sd = float(np.std(np.array(lo_r) - np.array(hi_r), ddof=1))
    t = diff / (sd / np.sqrt(len(lo_r))) if sd > 0 else np.nan
    rep.stats["liq_tilt_lo_ret"] = lo_m
    rep.stats["liq_tilt_hi_ret"] = hi_m
    rep.stats["liq_tilt_t"] = t

    detail = (f"按调仓日成交额分三组:最不流动 1/3 每期 {lo_m:+.3%}"
              f"（占权重 {np.mean(lo_w):.0%}）,"
              f"最流动 1/3 每期 {hi_m:+.3%}（占权重 {np.mean(hi_w):.0%}）"
              f"\n差 {diff:+.3%}/期,配对 t={t:.2f}（{len(lo_r)} 期）")
    if np.isfinite(t) and t >= 2.0:
        rep.add(WARN, "超额集中在最难成交的标的", detail,
                "放大规模时最先被迫放弃的恰好是赚钱的那批 ⇒ "
                "容量损失不是线性的,超出容量后 alpha 衰减会比"
                "冲击成本本身更快。做规模规划时不能只按成本外推",
                section=SECTION)
    else:
        rep.add(OK, "流动性集中度",
                detail + " —— 收益与流动性分组无系统性关联", section=SECTION)


# ---------------- ④ 规模衰减曲线 ----------------

def size_decay(rets: pd.Series, wm: pd.DataFrame, am: pd.DataFrame,
               ppy: float, sizes=SIZE_GRID) -> pd.DataFrame:
    """不同规模下的净年化:毛超额 − 冲击成本。

    冲击按平方根律 cost_bp ≈ K·sqrt(参与率),参与率 = 下单额/成交额。
    ★ 这是【形状】而非精确值：K 不可能对所有标的都准。
    """
    from .core import annualize
    wm2, am2 = wm.align(am, join="left", axis=1)
    am2 = am2.reindex(index=wm2.index).ffill()
    dates = list(wm2.index)
    dws, advs = [], []
    for prev, cur in zip(dates[:-1], dates[1:]):
        dw = (wm2.loc[cur] - wm2.loc[prev]).abs()
        tr = dw[dw > 1e-12]
        if tr.empty:
            continue
        adv = am2.loc[cur, tr.index]
        ok = np.isfinite(adv) & (adv > 0)
        if not ok.any():
            continue
        dws.append(tr[ok])
        advs.append(adv[ok])

    gross = annualize(rets, ppy)["ann_ret"]
    rows = []
    for S in sizes:
        # 每期的加权平均冲击成本（bp of 组合）
        costs = []
        for dw, adv in zip(dws, advs):
            part = (S * dw) / adv
            bp = IMPACT_K * np.sqrt(np.clip(part, 0, None))
            costs.append(float((dw * bp).sum()))     # dw 加权,单位 bp
        c_per_period = float(np.mean(costs)) * 1e-4 if costs else 0.0
        net = pd.Series(rets).astype(float) - c_per_period
        rows.append(dict(size=S,
                         impact_bp_per_period=float(np.mean(costs)) if costs else 0.0,
                         net_ann=annualize(net, ppy)["ann_ret"],
                         gross_ann=gross))
    return pd.DataFrame(rows).set_index("size")


def check_size_decay(rets: pd.Series, wm: pd.DataFrame, am: pd.DataFrame,
                     ppy: float, rep: AuditReport) -> None:
    """④ 规模衰减曲线:多大规模会把超额吃光。"""
    if am is None:
        rep.skip("规模衰减曲线", "价格面板没有成交额列")
        return
    if rets is None or len(rets) < 6:
        rep.skip("规模衰减曲线", "收益序列不足 6 期")
        return
    d = size_decay(rets, wm, am, ppy)
    if not len(d):
        rep.skip("规模衰减曲线", "没有可用的换手与成交额数据")
        return

    gross = float(d["gross_ann"].iloc[0])
    rep.stats["size_decay"] = {float(k): float(v)
                              for k, v in d["net_ann"].items()}
    # 净年化转负的第一个规模
    neg = d[d["net_ann"] <= 0]
    breakeven = float(neg.index.min()) if len(neg) else np.inf
    rep.stats["capacity_breakeven_cny"] = breakeven

    lines = [f"毛年化 {gross:.2%}；按平方根律冲击（K={IMPACT_K:.0f}）:"]
    for S, row in d.iterrows():
        lines.append(f"  规模 {_fmt_cny(S):>8s} ⇒ 冲击 "
                     f"{row['impact_bp_per_period']:.0f}bp/期、"
                     f"净年化 {row['net_ann']:+.2%}")
    detail = "\n".join(lines)
    detail += ("\n★ 这是【形状】不是精确值:冲击系数不可能对所有标的都准。"
               "看的是从哪个量级开始塌,不是某个规模的具体数字")

    if np.isfinite(breakeven):
        rep.add(WARN, f"规模到 {_fmt_cny(breakeven)} 超额归零", detail,
                f"超过 {_fmt_cny(breakeven)} 后这份策略不再有正超额。"
                "这不影响回测净值对不对,但决定了它能不能支撑一个产品 —— "
                "汇报规模时必须同时给容量",
                section=SECTION)
    else:
        rep.add(OK, "规模衰减曲线",
                detail + f"\n测试的最大规模 {_fmt_cny(max(SIZE_GRID))} 仍有正超额",
                section=SECTION)
