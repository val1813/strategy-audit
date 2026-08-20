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

from .report import BLOCK, SKIP, WARN, AuditReport

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
    for col in ("open", "amount", "up_limit", "down_limit"):
        if col in p.columns:
            p[col] = pd.to_numeric(p[col], errors="coerce")
    for col in ("is_suspended", "is_st"):
        if col in p.columns:
            if not pd.api.types.is_bool_dtype(p[col]):
                p[col] = p[col].map(lambda x: str(x).strip().lower() in
                                    {"1", "true", "yes", "y", "是"}
                                    if pd.notna(x) else np.nan)

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


# 自报净值与按权重+价格重算的净值，累计收益相对偏差门槛。
#
# ★ 门槛必须【不对称】—— 两个方向的含义完全不同
# ------------------------------------------------
#   自报 > 重算（rel < 0）  你的净值里有权重表【解释不了】的收益。
#                           权重表是策略的全部内容，它算不出你汇报的收益，
#                           就说明有一部分收益来自权重表之外 —— 这是
#                           要优先查的方向，门槛必须紧。
#   重算 > 自报（rel > 0）  你的回测比权重表【更保守】。最常见的原因就是
#                           扣了成本 —— 那是正确做法，不该报警。
#
# 用对称门槛会把「老老实实扣了成本」判成缺陷：单边 25bp、年换手 8x、
# 十年样本，累计差轻松超过 5%。这是最普通的正确实践，误报它等于
# 惩罚做对的人。所以良性方向改成先把缺口折算成【隐含每期成本】，
# 只有在这个数大到不像成本时才升级。
NAV_RECON_WARN = 0.01           # 危险方向：超过即 WARN
NAV_RECON_BLOCK = 0.03          # 危险方向：超过即 BLOCK
# 良性方向：缺口折算成单边每期 bp 后，超过这个数就不像「扣成本」了
NAV_RECON_COST_IMPLAUSIBLE_BP = 100.0


def check_nav_reconciliation(rets: pd.Series, own: pd.Series, role: str,
                             rep: AuditReport,
                             turnover: float | None = None) -> dict:
    """★ 拿你【自报的净值】与按权重+价格【重算的净值】对账。

    这是本工具最核心的一项：宣称审的就是「回测到净值这一段」，
    那么当两份东西同时在手时，第一件该做的事就是问它们是否一致。

    实测（盲测一份四因子月频策略）：自报年化 6.19%、重算 6.66%，
    累计差 6.0%。原因是作者在月内每天把权重 ffill 回等权 ——
    等于隐含的日频再平衡，而权重表里【看不到】这件事。
    没有这一项，工具会拿自己重算的曲线去审，然后对 6% 的缺口一言不发：
    所有检查都算在一条【客户从未汇报过】的净值上，而报告照样打印得
    像模像样。这正是本工具反复强调要避免的失败模式。

    偏差的方向也有信息：
        重算 > 自报   你的回测比权重表更保守（如扣了成本、或有现金拖累）
        重算 < 自报   你的净值里有权重表【解释不了】的收益 —— 优先查这个
    """
    r = pd.Series(rets).dropna().astype(float)
    o = pd.Series(own).dropna().astype(float)
    if len(r) < 3 or len(o) < 3:
        rep.skip("自报净值对账", "序列过短，无法对账")
        return {}

    # ★ 必须在【同一个窗口】上比，不能盲取首末。
    # 净值序列有两种常见写法，从数值上无法区分：
    #     起点锚在 1.0（第一天净值 = 1）
    #     起点就是第一期结束后的值（第一期收益已经算进起点）
    # 盲取首末会在后一种写法上凭空报出一期收益的偏差 —— 干净策略
    # 因此被报 BLOCK。误报比漏报更糟：报警一旦不可信就没人看了。
    #
    # r 的索引是各调仓日 t_0..t_{n-1}，其中 t_k 那一期跨 [t_k, t_{k+1}]。
    # 所以用自报曲线在 [t_0, t_{n-1}] 上的增长，对比【前 n-1 期】的累乘 ——
    # 两边严格覆盖同一段时间。
    if not isinstance(r.index, pd.DatetimeIndex) or \
            not isinstance(o.index, pd.DatetimeIndex):
        rep.skip("自报净值对账", "索引不是日期，无法对齐到同一窗口")
        return {}

    # 取【两条曲线共同覆盖】的调仓日窗口 [a, b]。
    # ★ 不能假设自报曲线正好从第一个调仓日开始：实测日频净值常常
    # 起于调仓日的次日（月末是周末时更是必然），盲取首末会直接判无法对账。
    o = o.sort_index()
    reb = r.index
    inside = reb[(reb >= o.index.min()) & (reb <= o.index.max())]
    if len(inside) < 3:
        rep.skip("自报净值对账",
                 f"自报曲线（{o.index.min().date()} ~ {o.index.max().date()}）"
                 f"与调仓日（{reb.min().date()} ~ {reb.max().date()}）"
                 f"重叠不足 3 期，无法对账")
        return {}
    a, b = inside.min(), inside.max()

    # Source-identity guard: a reconciliation made from the same numbers is
    # not independent evidence. Compare period-by-period, not only endpoints,
    # because two different paths can have the same cumulative return.
    if role == "nav":
        pts = []
        for d in inside:
            j = o.index.asof(d)
            if not pd.isna(j) and (not pts or j != pts[-1]):
                pts.append(j)
        own_period = o.loc[pts].pct_change().dropna() if len(pts) >= 2 else pd.Series(dtype=float)
        # A NAV stamped at period end naturally aligns its change with the end
        # date. This also handles the common nav=(1+ret).cumprod() export.
        mine_period = r.reindex(own_period.index)
    else:
        common = o.index.intersection(r.index)
        own_period = o.reindex(common)
        mine_period = r.reindex(common)
    valid = own_period.notna() & mine_period.notna()
    max_resid = (float((own_period[valid] - mine_period[valid]).abs().max())
                 if int(valid.sum()) >= 2 else np.nan)
    rep.stats["nav_recon_max_period_resid"] = max_resid
    if np.isfinite(max_resid) and max_resid < 1e-9:
        detail = ("你交的净值与按权重+价格重算的结果在浮点精度内完全一致"
                  f"（max 残差 {max_resid:.1e}）。这说明两者由同一批数字算出；"
                  "平台导出件通常如此。这是同源恒等式，本项因此不可审，"
                  "不是审过通过了。")
        rep.add(SKIP, "自报净值对账：本项不可审（同源）", detail,
                "要让这一项有内容，需要一份独立来源的价格面板",
                section="输入契约")
        return dict(rel=0.0, max_resid=max_resid, identity=True)

    if role == "nav":
        # asof：调仓日那天自报曲线上不一定有点（停牌/非交易日）
        i0, i1 = o.index.asof(a), o.index.asof(b)
        if pd.isna(i0) or pd.isna(i1) or i0 >= i1 or o.loc[i0] == 0:
            rep.skip("自报净值对账", "自报曲线在对账窗口内取不到有效端点")
            return {}
        own_cum = float(o.loc[i1] / o.loc[i0])
    else:
        seg = o.loc[(o.index > a) & (o.index <= b)]
        if len(seg) < 2:
            rep.skip("自报净值对账", "自报收益率序列在对账窗口内点数不足")
            return {}
        own_cum = float((1.0 + seg).prod())
    # 与 own 严格同窗：第 k 期覆盖 [t_k, t_{k+1}]，所以取 a ≤ t_k < b
    mine_cum = float((1.0 + r.loc[(r.index >= a) & (r.index < b)]).prod())
    if not (np.isfinite(own_cum) and np.isfinite(mine_cum)) or own_cum <= 0:
        rep.skip("自报净值对账", "累计增长非有限值，无法对账")
        return {}
    rep.stats["nav_recon_window"] = f"{a.date()} ~ {b.date()}"

    rel = mine_cum / own_cum - 1.0
    rep.stats["nav_recon_own_cum"] = own_cum
    rep.stats["nav_recon_recomputed_cum"] = mine_cum
    rep.stats["nav_recon_rel"] = rel

    detail = (f"你自报的累计 {own_cum - 1:+.2%}，"
              f"按权重+价格重算 {mine_cum - 1:+.2%}"
              f"（相对偏差 {rel:+.2%}）")
    if abs(rel) <= NAV_RECON_WARN:
        rep.add("OK", "自报净值对账",
                detail + f" —— 两条曲线一致（≤{NAV_RECON_WARN:.0%}），"
                "下面各项算的就是你汇报的那条净值",
                section="输入契约")
        return dict(rel=rel)

    n_per = int(((r.index >= a) & (r.index < b)).sum())

    if rel > 0:
        # ---- 良性方向：重算【高于】自报，最常见的原因就是你扣了成本 ----
        # 把缺口折算成「隐含单边每期成本」，看这个数像不像成本。
        # 需要换手才能折算；没有换手信息时只报缺口，不升级。
        # ★ 换手必须【显式传进来】，不能读 rep.stats。
        # 本项在输入契约阶段跑（族一还没跑），那时 stats 里还没有换手 ——
        # 读它只会永远拿到 None，于是良性方向永远升级成 WARN。
        # 这类「静默拿不到值」的依赖是最难发现的：报告照样打印得像模像样。
        to = turnover
        implied_bp = np.nan
        if to and n_per > 0 and to > 0:
            # (1+g)^n ≈ 缺口 ⇒ 每期缺口 ≈ rel/n；单边 bp = 每期缺口/(2·τ)
            implied_bp = (rel / n_per) / (2.0 * to) * 1e4
        if np.isfinite(implied_bp) and implied_bp <= NAV_RECON_COST_IMPLAUSIBLE_BP:
            rep.add("OK", "自报净值低于重算（像是扣了成本）",
                    detail + f"\n折算成隐含单边成本约 {implied_bp:.1f}bp/笔"
                    f"（{n_per} 期、单边换手 {to:.1%}/期）",
                    "重算是【毛】口径（不扣成本），你自报的更低 —— 这与"
                    "「回测里扣了交易成本」一致，是正确做法，不是缺陷。"
                    "下面各项按【毛】口径计算，所以看到的超额比你汇报的高，"
                    "盈亏平衡成本那一项才是可以直接和这个数对照的",
                    section="输入契约")
            return dict(rel=rel, implied_cost_bp=implied_bp)
        if np.isfinite(implied_bp):
            extra = (f"\n折算成隐含单边成本 {implied_bp:.0f}bp/笔 —— "
                     f"这个数大到不像交易成本"
                     f"（门槛 {NAV_RECON_COST_IMPLAUSIBLE_BP:.0f}bp）")
            why = ("重算【高于】自报。若这是扣成本造成的，量级偏大；"
                   "也可能是留了现金、月内有再平衡（权重表只记了调仓日的"
                   "目标）、或收益口径不一致。")
        else:
            # ★ 算不出隐含成本时不许暗示「量级偏大」——
            # 那是我们没能算，不是它真的偏大。把不确定性说出来。
            extra = "\n无换手信息，无法折算成隐含成本"
            why = ("重算【高于】自报，方向与「回测里扣了成本」一致 —— "
                   "但没有换手信息，无法判断量级像不像成本。"
                   "若你确实扣了成本，这一条预期之内；否则请查现金、"
                   "月内再平衡、或收益口径。")
        rep.add(WARN, f"自报净值比重算低 {abs(rel):.1%}", detail + extra,
                why + "下面各项算的是【重算】那条曲线（毛口径）",
                section="输入契约")
        return dict(rel=rel, implied_cost_bp=implied_bp)

    # ---- 危险方向：自报【高于】重算 ⇒ 权重表解释不了你汇报的收益 ----
    lvl = BLOCK if abs(rel) > NAV_RECON_BLOCK else WARN
    rep.add(lvl, f"自报净值比重算高 {abs(rel):.1%}", detail,
            "★ 权重表【算不出】你汇报的收益 —— 而权重表本应是策略的全部内容。"
            "优先查这一项：权重表之外的择时/杠杆、调仓日错位（用了 t 日"
            "收盘定权重却吃了 t 日涨幅）、或收益计算口径错误。"
            "注意方向：这个缺口不可能由「扣成本」解释，扣成本只会让"
            "自报【低于】重算。下面各项算的是【重算】那条曲线",
            section="输入契约")
    return dict(rel=rel)
