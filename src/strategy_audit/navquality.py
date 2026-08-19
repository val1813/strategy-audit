"""族七：净值质量 —— 只要一条净值曲线就能跑。

审的是「这条曲线本身像不像一条真实成交出来的净值」，三项：

    ① 平滑与滞后定价   收益自相关 ⇒ 波动被低估、Sharpe 被高估
    ② 停滞估值         零收益/重复净值占比 ⇒ 估值没更新
    ③ 期末修饰         期末最后一日收益是否系统性高于其它日

★ 为什么要有这一族
----------------
前六族里门槛最低的族三也只审「显著性」，它默认那条曲线是真的。
但机构拿到的第一份东西往往【只有净值】—— 没有权重、没有价格。
那时能审的只有 4 项，而这 4 项全都建立在「这条曲线本身可信」之上。

这一族补的就是那个前提：不问策略好不好，问这条曲线像不像
真实成交出来的。三项都只需要一条序列，所以适用面最广。

★ 门槛必须随样本量缩放
--------------------
实测（2000 次纯噪声）：AC1 零分布的标准差几乎正好是 1/√n ——
    n=60   sd 0.124（1/√n = 0.129）  p95 +0.193
    n=120  sd 0.091（1/√n = 0.091）  p95 +0.141
    n=250  sd 0.063（1/√n = 0.063）  p95 +0.102
所以固定门槛（比如「AC1 > 0.2 才报」）在长样本上会漏、在短样本上会误报。
判据用 t = AC1 · √n，与样本量无关。

★ 这一族只报事实，不下「造假」的结论
--------------------------------
自相关高有多种成因：真的平滑了、持仓含流动性差的资产（估值天然滞后）、
或者月频数据本身的重叠。工具给量级和它对 Sharpe 的影响，
判断留给用户 —— 「这条净值是伪造的」不是审计工具能下的结论。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .core import annualize
from .report import OK, WARN, AuditReport

SECTION = "净值质量"

# 自相关的 t 门槛（t = AC1·√n）。★ 见模块头：零分布 sd ≈ 1/√n，
# 所以 t 就是标准化后的偏离，2.0/3.0 对应约 p95/p99。
AC1_T_WARN = 2.0
AC1_T_BLOCK = 3.0

# 解平滑后波动放大超过这个倍数 ⇒ Sharpe 的高估已经不能忽略
UNSMOOTH_WARN = 1.15

# 零收益（净值原地不动）占比超过这个比例 ⇒ 估值可能没更新。
# ★ 连续分布下零收益的概率是 0，实测 500 次纯噪声全为 0.0000，
# 所以任何显著的占比都值得报。5% 已经很宽容 —— 留给真实存在的
# 停牌日和节假日错位。
STALE_WARN = 0.05
STALE_BLOCK = 0.30

# 期末修饰：期末最后一日收益减去其它日均值，折算成 t
DRESS_T_WARN = 2.5

MIN_N = 24          # 少于这么多点，这一族的每一项都没有区分力


def ac1(x) -> float:
    """一阶自相关。"""
    v = np.asarray(x, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) < 3:
        return float("nan")
    # ★ 极端量级（1e300 之类）下 v*v 会溢出成 inf，于是比值变 nan
    # 并伴随 RuntimeWarning。先按尺度归一，把溢出挡在算式之外 ——
    # 自相关是无量纲量，除以尺度不改变结果。
    v = v - v.mean()
    scale = float(np.max(np.abs(v))) if len(v) else 0.0
    if not np.isfinite(scale) or scale <= 0:
        return float("nan")
    v = v / scale
    with np.errstate(over="ignore", invalid="ignore"):
        den = float((v * v).sum())
        if not np.isfinite(den) or den <= 0:
            return float("nan")
        out = float((v[:-1] * v[1:]).sum() / den)
    return out if np.isfinite(out) else float("nan")


def unsmooth_amplification(r) -> tuple[float, float]:
    """Getmansky-Lo-Makarov 式一阶解平滑：波动被低估了多少倍。

    设观测收益是真实收益的移动平均  r_obs(t) = θ0·r(t) + θ1·r(t−1)，
    θ0 + θ1 = 1（不改变均值，只搬移方差）。则

        AC1(r_obs) = θ0·θ1 / (θ0² + θ1²)
        sd(r) / sd(r_obs) = 1 / sqrt(θ0² + θ1²)

    由观测到的 AC1 数值反解 θ1 ∈ (0, 0.5)，给出放大倍数。

    ★ 只做一阶。GLM 原文用 q 阶 MA 并做极大似然，这里刻意退化成
    一阶闭式解：审计要的是「Sharpe 被高估了多少」的量级，
    而高阶拟合会引入模型选择自由度 —— 那正是本工具反对的东西。
    一阶给出的是【下界】（真实平滑阶数更高时放大更多）。

    返回 (放大倍数, θ1)。AC1 ≤ 0 时返回 (1.0, 0.0)。
    """
    a = ac1(r)
    if not np.isfinite(a) or a <= 0:
        return 1.0, 0.0
    xs = np.linspace(1e-4, 0.4999, 4000)
    v = xs * (1.0 - xs) / ((1.0 - xs) ** 2 + xs ** 2)
    x = float(xs[int(np.argmin(np.abs(v - a)))])
    return float(1.0 / np.sqrt((1.0 - x) ** 2 + x ** 2)), x


def stale_share(nav_or_ret: pd.Series, role: str) -> float:
    """净值原地不动的比例（估值没更新的直接症状）。

    ★ 同族内其它入口：dropna() 拦不住 ±inf，必须用 isfinite。
    """
    v = pd.Series(nav_or_ret).astype(float)
    v = v[np.isfinite(v)]
    if len(v) < 3:
        return float("nan")
    d = v.diff().dropna().values if role == "nav" else v.values
    return float((np.abs(d) < 1e-12).mean())


# ---------------- ① 平滑与滞后定价 ----------------

def check_smoothing(rets: pd.Series, ppy: float, rep: AuditReport) -> dict:
    """① 收益自相关 ⇒ 波动被低估、Sharpe 被高估。"""
    # ★ ±inf 必须在入口就剔掉，不能只 dropna()。
    # 敌意输入测试抓到的：序列含一个 inf 时 AC1 仍算得出有限值，
    # 但 Sharpe 变成 nan，于是报告打出「Sharpe nan → nan」还给 OK 结论。
    # dropna() 拦不住 inf —— 这是 pandas 的老坑，本模块每个入口都要自己防。
    r = pd.Series(rets).astype(float)
    r = r[np.isfinite(r)]
    n = len(r)
    if n < MIN_N:
        rep.skip("平滑与滞后定价", f"只有 {n} 个有限值的点（需要 ≥{MIN_N}）")
        return {}

    a = ac1(r.values)
    # ★ 算不出自相关就必须跳过，不能带着 nan 往下走。
    # 敌意输入测试抓到的：序列含 ±inf 时 ac1 返回 nan，于是报告打出
    # 「收益一阶自相关 +nan（58 期，t=+nan）」并照样给出 OK 结论 ——
    # 一份看起来像模像样、实际什么都没审的报告。
    # 上游 _clean_series 会清掉 ±inf，但本函数是公开 API，必须自己站得住。
    if not np.isfinite(a):
        rep.skip("平滑与滞后定价",
                 "自相关无法计算（序列可能含 ±inf 或方差为 0）")
        return {}
    t = a * np.sqrt(n)
    amp, th1 = unsmooth_amplification(r.values)
    sr = annualize(r, ppy)["sharpe"]
    # ★ Sharpe 为负时除以放大倍数会让它【变好】（−0.24 → −0.17）——
    # 那不是「修正」，那是把一个亏钱的策略修饰成亏得少一点。
    # 解平滑放大的是【波动】，所以正确写法是 μ/(σ·amp)：
    # 分母变大时 |Sharpe| 必然变小，符号不变。负 Sharpe 因此向 0 移动
    # 是数学事实，但那不该被说成「修正后更真实」。
    # 报告只在 Sharpe 为正时把它当成折扣，为负时明说方向。
    sr_adj = sr / amp if (amp > 0 and np.isfinite(sr)) else sr
    rep.stats.update(nav_ac1=a, nav_ac1_t=t, nav_unsmooth_amp=amp,
                     nav_sharpe_reported=sr, nav_sharpe_unsmoothed=sr_adj)

    detail = (f"收益一阶自相关 {a:+.3f}（{n} 期，t={t:+.2f}）"
              f"\n解平滑后波动放大 {amp:.2f}x ⇒ "
              f"Sharpe {sr:.2f} → {sr_adj:.2f}"
              f"\n★ 门槛随样本量缩放：零分布标准差 ≈ 1/√n = {1/np.sqrt(n):.3f}，"
              f"所以判据是 t 而不是固定的相关系数")

    # ★ 必须【同时】统计显著且经济上重要才报警。
    # 实测：盲测那份日频真实净值 n=2645、AC1=+0.048 ⇒ t=2.47 统计显著,
    # 但解平滑放大只有 1.05x —— Sharpe 从 0.29 变 0.28,没有任何决策含义。
    # 长样本下极小的自相关必然「显著」（t = AC1·√n 随 n 增长）,
    # 只看 t 就会把每一条日频净值都报成可疑。
    # 反过来只看放大倍数则在短样本上误报（噪声也能凑出 1.2x,见零分布 p95）。
    # 两个条件取【与】：既排除长样本的鸡毛蒜皮,也排除短样本的噪声。
    if t < AC1_T_WARN or amp < UNSMOOTH_WARN:
        why = ("自相关与噪声不可区分" if t < AC1_T_WARN else
               f"自相关虽显著（t={t:+.2f}）,但解平滑放大仅 {amp:.2f}x,"
               f"对 Sharpe 的影响（{sr:.2f}→{sr_adj:.2f}）无决策含义")
        rep.add(OK, "平滑与滞后定价", detail + f"\n—— {why}",
                section=SECTION)
        return dict(ac1=a, t=t, amp=amp)

    # ★ 本族不出 BLOCK。自相关高有多种正当成因（含估值滞后的资产、
    # 数据重叠），把它做成 BLOCK 会逼用户为了过闸门去改一条
    # 可能本来就没错的净值。强度通过 t 与放大倍数表达。
    strong = t >= AC1_T_BLOCK
    # ★ Sharpe 为负时不许说「Sharpe 被高估」——
    # 解平滑放大波动，负 Sharpe 会向 0 移动（−0.24 → −0.17），
    # 那不是「高估被修正」。此时该说的是波动被低估。
    if np.isfinite(sr) and sr > 0:
        impact = (f"自相关为正会让波动被低估、Sharpe 被高估约 {amp:.2f} 倍。"
                  f"对外披露的 Sharpe 应按解平滑后的 {sr_adj:.2f} 计，"
                  f"风险预算也要按放大后的波动编。")
    else:
        impact = (f"自相关为正会让波动被低估约 {amp:.2f} 倍。"
                  f"本样本 Sharpe 本就不为正（{sr:.2f}），"
                  f"所以这里要修正的是【风险】而不是收益："
                  f"真实波动更大、回撤更深，风险预算须按放大后的波动编。")
    rep.add(WARN, "净值可能被平滑或滞后定价"
            + ("（自相关很强）" if strong else ""), detail,
            impact + "成因不止一种：真的做了平滑、持仓含估值天然滞后的资产、"
            "或数据本身重叠 —— 工具给量级，不下结论",
            section=SECTION)
    return dict(ac1=a, t=t, amp=amp)


# ---------------- ② 停滞估值 ----------------

def check_stale(nav_or_ret: pd.Series, role: str, rep: AuditReport) -> dict:
    """② 净值原地不动的比例。"""
    v = pd.Series(nav_or_ret).astype(float)
    v = v[np.isfinite(v)]
    if len(v) < MIN_N:
        rep.skip("停滞估值", f"只有 {len(v)} 个有限值的点（需要 ≥{MIN_N}）")
        return {}
    s = stale_share(v, role)
    if not np.isfinite(s):
        rep.skip("停滞估值", "无法计算零变动占比")
        return {}
    rep.stats["nav_stale_share"] = s

    detail = (f"{s:.1%} 的相邻点净值完全没变（{len(v)} 个点）"
              f"\n★ 连续分布下零收益的概率是 0："
              f"实测 500 次纯噪声全为 0.0%，所以任何显著占比都值得看")
    if s < STALE_WARN:
        rep.add(OK, "停滞估值", detail + " —— 占比正常", section=SECTION)
        return dict(stale=s)
    rep.add(WARN, "净值有大量原地不动的点", detail,
            f"{s:.1%} 的点净值没有变化。常见成因：估值按周/月更新但曲线"
            f"按日给出（⇒ 日频 Sharpe 无意义，应按真实估值频率重算）、"
            f"停牌期间挂上一日价、或数据被前向填充过。"
            f"{'占比之大已经说明这不是日频净值。' if s >= STALE_BLOCK else ''}"
            f"请按【真实估值频率】重算年化与 Sharpe",
            section=SECTION)
    return dict(stale=s)


# ---------------- ③ 期末修饰 ----------------

def check_period_end_dressing(rets: pd.Series, rep: AuditReport,
                              freq: str = "ME") -> dict:
    """③ 期末最后一日的收益是否系统性高于其它日。

    ★ 这一项【只在日频序列上有意义】。月频序列每个点本身就是月末，
    没有「期末 vs 期内」的区分 —— 那时必须跳过而不是硬算。
    """
    r = pd.Series(rets).astype(float)
    r = r[np.isfinite(r)]
    if not isinstance(r.index, pd.DatetimeIndex):
        rep.skip("期末修饰", "索引不是日期，无法定位期末")
        return {}
    if len(r) < MIN_N * 3:
        rep.skip("期末修饰", f"只有 {len(r)} 个点（需要 ≥{MIN_N * 3}）")
        return {}

    grp = r.groupby([r.index.year, r.index.month])
    if grp.size().median() < 5:
        rep.skip("期末修饰",
                 "每月不足 5 个点（月频或更低频序列没有期末/期内之分）")
        return {}

    last_idx = grp.tail(1).index
    is_last = r.index.isin(last_idx)
    a, b = r[is_last], r[~is_last]
    if len(a) < 6 or len(b) < 6:
        rep.skip("期末修饰", "期末或期内点数不足")
        return {}

    diff = float(a.mean() - b.mean())
    se = float(np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b)))
    t = diff / se if se > 0 else np.nan
    rep.stats.update(nav_dressing_bp=diff * 1e4, nav_dressing_t=t)

    detail = (f"月末最后一个交易日平均 {a.mean() * 1e4:+.1f}bp、"
              f"其它日 {b.mean() * 1e4:+.1f}bp，"
              f"差 {diff * 1e4:+.1f}bp（t={t:+.2f}，{len(a)} 个月末点）")
    if not np.isfinite(t) or abs(t) < DRESS_T_WARN:
        mde = 1.96 * se * 1e4
        rep.add(OK, "期末修饰", detail
                + f"\n—— 与噪声不可区分（本样本最小可检出 {mde:.1f}bp）",
                section=SECTION)
        return dict(diff=diff, t=t)
    rep.add(WARN, "月末收益系统性偏高", detail,
            f"月末最后一日比其它日高 {diff * 1e4:+.1f}bp 且显著。"
            f"这在按月披露的产品里值得注意（拉抬收盘、估值择时），"
            f"但也可能是真实的月末效应（指数调仓、资金面）。"
            f"工具给量级不下结论 —— 若这是真实交易，"
            f"请确认它在样本外也存在",
            section=SECTION)
    return dict(diff=diff, t=t)
