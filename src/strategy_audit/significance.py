"""族三：策略层显著性。只要一条净值/收益率曲线就能跑。

四项：

    ① 年份集中度   超额是否集中在极少数年份
    ② NW lag 敏感  t 值是否靠 lag 选择撑起来
    ③ 多重检验折扣 按试过的配置数折扣后还剩多少
    ④ 回撤与极值   最大回撤、恢复期、单期极值贡献

★ 为什么这一族门槛最低但排在报告最后
--------------------------------
门槛低是因为只要曲线；排最后是因为它审的是「这个数字可不可信」，
而前两族审的是「这个数字是不是真的」。净值本身偷了分的时候，
再精确的显著性检验也只是把假结论算得更精确。

★ 本族的每一项都只报事实，不下「有效/无效」的结论
----------------------------------------------
「61% 的超额来自两个年份」是事实；「所以策略无效」是判断，
取决于用户能不能承受那种收益分布。工具给事实和量级。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .core import annualize
from .report import BLOCK, OK, WARN, AuditReport

SECTION = "策略层显著性"

# 集中度按【置换零分布的分位】判，不用固定比例门槛。
# ★ 实测（400 次纯噪声模拟，10 年月频）：固定门槛 0.50 的误报率 61%，
# 而零分布的中位数本身就是 0.53 —— 门槛定在中位数以下，等于一半的
# 干净策略都被报警。集中度随年数变化，固定数字不可能对。
CONC_PCTILE_WARN = 0.95
CONC_PCTILE_BLOCK = 0.99
N_PERM = 600
PERM_SEED = 20260812      # 固定种子：同一份输入必须给同一份报告

# 少于这么多年度时集中度检验没有区分力（零分布中位数会顶到 100%）
MIN_YEARS_FOR_CONC = 5

# NW lag 从 0 扫到 N，t 的最大/最小比值超过这个 ⇒ 结论对 lag 敏感
NW_RATIO_WARN = 1.5
NW_RATIO_BLOCK = 2.0

# 跨过 |t|=2 时，最小要跨多远才算「lag 挑出来的显著性」。
# ★ 实测 200 次纯噪声：16% 会跨过 |t|=2，跨度中位数 0.57、p90 0.91。
# 所以 1.0 以下基本都是贴线晃动，定在 1.0 才能把噪声挡掉。
NW_CROSS_MIN_GAP = 1.0

# 最好一期偏离均值多少个标准差 ⇒ 视为离群点（可能是数据错误）。
# ★ 不用「占累计收益的比例」当门槛：那个量在累计收益接近 0 时分母趋零、
# 会爆到 >1，实测纯噪声 p90=0.78 / p95=1.45，用 0.30 误报率 27%。
# 4σ 在 120 期正态样本里的期望出现次数约 0.004 次 —— 出现就值得看一眼。
SINGLE_PERIOD_Z = 4.0


def to_returns(s: pd.Series, role: str) -> pd.Series:
    """把净值或收益率统一成【收益率】。"""
    v = pd.Series(s).dropna().astype(float).sort_index()
    if role == "nav":
        return v.pct_change().dropna()
    return v


def newey_west_t(x: np.ndarray, lag: int) -> float:
    """均值为 0 的 Newey-West t 统计量。"""
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 3:
        return float("nan")
    mu = x.mean()
    e = x - mu
    gamma0 = float(e @ e) / n
    var = gamma0
    for L in range(1, min(lag, n - 1) + 1):
        w = 1.0 - L / (lag + 1.0)
        cov = float(e[L:] @ e[:-L]) / n
        var += 2.0 * w * cov
    if var <= 0:
        return float("nan")
    return mu / np.sqrt(var / n)


def check_year_concentration(rets: pd.Series, rep: AuditReport,
                             bench: pd.Series | None = None) -> None:
    """① 超额是否集中在极少数年份。

    ★ 实测过的案例：61% 的超额来自两个年份 ⇒ 不可用。
    年化数字会把这件事完全藏起来。
    """
    r = pd.Series(rets).dropna().astype(float)
    if bench is not None:
        b = pd.Series(bench).reindex(r.index).astype(float)
        r = (r - b).dropna()
    if len(r) < 24 or not isinstance(r.index, pd.DatetimeIndex):
        rep.skip("年份集中度", "序列不足 24 期或索引不是日期")
        return

    by_year = r.groupby(r.index.year).apply(lambda x: float((1 + x).prod() - 1))
    total = float((1 + r).prod() - 1)
    rep.stats["year_returns"] = by_year.to_dict()

    # ★ 少于 5 个年度时这一项【没有意义】，必须跳过而不是报 OK。
    # 实测：3 个年度时置换零分布的中位数就是 100%（最好两年当然占满
    # 全部正收益），于是任何策略都「与随机打散不可区分」——
    # 报 OK 等于给了一个假的通过。
    if len(by_year) < MIN_YEARS_FOR_CONC:
        rep.skip("年份集中度",
                 f"只覆盖 {len(by_year)} 个年度（需要 ≥{MIN_YEARS_FOR_CONC}）："
                 f"年数太少时「最好两年占满正收益」是必然，检验无区分力")
        return

    pos = by_year[by_year > 0]
    if total <= 0 or pos.empty:
        rep.add(WARN, "累计超额不为正",
                f"{len(by_year)} 个年度累计 {total:+.2%}"
                f"（正收益年份 {len(pos)}/{len(by_year)}）",
                "集中度对负收益策略没有意义 —— 先解决方向问题",
                section=SECTION)
        return

    top = pos.nlargest(min(2, len(pos)))
    share = float(top.sum() / pos.sum())
    yrs = "、".join(f"{int(y)}年 {v:+.1%}" for y, v in top.items())

    # ★ 用置换零分布定标：把同一批收益随机打散到各年，看「最好两年占比」
    # 本来能有多高。这样门槛自动随年数、波动率调整。
    null = _conc_null(r)
    pct = float((null < share).mean()) if len(null) else np.nan
    rep.stats["year_conc_share"] = share
    rep.stats["year_conc_pctile"] = pct

    detail = (f"{len(by_year)} 个年度，累计 {total:+.2%}。"
              f"最好的 {len(top)} 个年度（{yrs}）"
              f"占全部正收益的 {share:.0%}")
    if len(null):
        detail += (f"\n置换零分布：随机打散同一批收益，该占比中位数 "
                   f"{np.median(null):.0%}、95 分位 {np.quantile(null, .95):.0%}"
                   f" ⇒ 实测落在 {pct:.0%} 分位")

    lvl = (BLOCK if pct >= CONC_PCTILE_BLOCK else
           WARN if pct >= CONC_PCTILE_WARN else OK)
    if lvl is OK:
        rep.add(OK, "年份集中度",
                detail + "\n—— 集中程度与「随机打散」不可区分，"
                "不构成集中度问题", section=SECTION)
        return
    n_neg = int((by_year <= 0).sum())
    rep.add(lvl, "超额集中在极少数年份", detail,
            f"{len(by_year)} 年里有 {n_neg} 年不赚钱，"
            f"且集中程度超过 {pct:.0%} 的随机打散情形。"
            "年化数字会把这件事完全藏起来 —— 若那 "
            f"{len(top)} 年是特定行情（如小盘股风格年），"
            "策略在别的年份就是不工作。请按年度而非全样本汇报",
            section=SECTION)


def _conc_null(r: pd.Series) -> np.ndarray:
    """集中度的置换零分布：保持收益值不变，只打散它们落在哪一年。"""
    v = r.values.astype(float)
    years = np.asarray(r.index.year)
    uniq = np.unique(years)
    if len(uniq) < 3:
        return np.array([])
    rng = np.random.default_rng(PERM_SEED)
    out = []
    for _ in range(N_PERM):
        perm = rng.permutation(v)
        by = np.array([float(np.prod(1.0 + perm[years == y]) - 1.0)
                       for y in uniq])
        pos = by[by > 0]
        if len(pos) < 2 or float(np.prod(1.0 + perm) - 1.0) <= 0:
            continue
        k = min(2, len(pos))
        out.append(float(np.sort(pos)[-k:].sum() / pos.sum()))
    return np.array(out)


def check_nw_lag_sensitivity(rets: pd.Series, rep: AuditReport,
                             max_lag: int = 12) -> None:
    """② t 值对 NW lag 选择的敏感度。

    ★ 实测过 lag 选择把 t 从 1.92 抬到 3.71 —— 跨过显著性门槛。
    报告必须把整条 lag 曲线的范围给出来，而不是只报一个 t。
    """
    r = pd.Series(rets).dropna().astype(float)
    if len(r) < 12:
        rep.skip("NW lag 敏感性", "序列不足 12 期")
        return

    lags = list(range(0, min(max_lag, len(r) // 3) + 1))
    ts = {L: newey_west_t(r.values, L) for L in lags}
    ts = {L: t for L, t in ts.items() if np.isfinite(t)}
    if len(ts) < 3:
        rep.skip("NW lag 敏感性", "t 统计量无法在多个 lag 下计算")
        return

    rep.stats["nw_t_by_lag"] = ts
    vals = np.array(list(ts.values()))
    t_lo, t_hi = float(vals.min()), float(vals.max())
    L_lo = min(ts, key=lambda k: ts[k])
    L_hi = max(ts, key=lambda k: ts[k])
    ratio = abs(t_hi) / abs(t_lo) if abs(t_lo) > 1e-9 else np.inf

    detail = (f"lag 0~{max(ts)} 扫描：t 从 {t_lo:.2f}（lag={L_lo}）"
              f"到 {t_hi:.2f}（lag={L_hi}）")

    # 跨过 |t|=2 且【跨得够远】才算问题。
    # ★ 实测（200 次纯噪声，120 期月频）：16% 的干净序列会在 lag 扫描中
    # 跨过 |t|=2，跨度中位数仅 0.57 —— 那是噪声，不是 lag 挑选的证据。
    # 第一版一跨过就 BLOCK ⇒ 六分之一的干净策略被判「不可信」。
    # 门槛：贴着 2.0 两侧晃（跨度 < NW_CROSS_MIN_GAP）只提示，不定罪。
    crosses = (abs(t_lo) < 2.0 <= abs(t_hi))
    gap = abs(t_hi) - abs(t_lo)
    if crosses and gap >= NW_CROSS_MIN_GAP:
        rep.add(WARN, "显著性随 NW lag 翻转", detail,
                f"|t|=2 这条线被 lag 选择跨过了（跨度 {gap:.2f}）—— "
                f"lag={L_lo} 时不显著、lag={L_hi} 时显著。"
                "请预先声明 lag 规则（如 Newey-West 自动带宽）并报告全区间，"
                "否则读者无法判断这个 t 是数据给的还是挑出来的",
                section=SECTION)
        return
    if crosses:
        rep.add(OK, "NW lag 敏感性",
                detail + f"\n虽跨过 |t|=2，但跨度仅 {gap:.2f} —— "
                "实测 16% 的纯噪声序列也会这样（跨度中位数 0.57），"
                "属于贴线晃动而非 lag 挑选",
                section=SECTION)
        return

    # ★ 比值大不等于有问题：t 从 3.30 到 6.93 比值 2.10，但两端都
    # 远超 |t|=2，结论根本没变 —— 第一版按比值 BLOCK 了这种情形。
    # lag 敏感性真正要紧的只有一件事：**结论会不会随 lag 翻转**。
    # 两端同侧（都显著或都不显著）时只提示比值，不定罪。
    both_signif = abs(t_lo) >= 2.0 and abs(t_hi) >= 2.0
    both_insignif = abs(t_lo) < 2.0 and abs(t_hi) < 2.0
    if both_signif or both_insignif:
        where = "两端都显著" if both_signif else "两端都不显著"
        note = (f"，最大/最小比 {ratio:.2f}（{where}，结论不随 lag 变）"
                if ratio > NW_RATIO_WARN else
                f"，最大/最小比 {ratio:.2f} —— 结论不靠 lag 撑")
        rep.add(OK, "NW lag 敏感性", detail + note, section=SECTION)
        return

    rep.add(WARN, "t 值对 lag 选择敏感",
            detail + f"，最大/最小比 {ratio:.2f}",
            "实测过 lag 选择把 t 从 1.92 抬到 3.71。"
            "报单个 t 值时必须同时说明 lag 怎么定的，"
            "否则读者无法判断这个 t 是数据给的还是挑出来的",
            section=SECTION)


def deflated_sharpe(sharpe: float, n_obs: int, n_trials: int,
                    ppy: float) -> dict:
    """按尝试次数折扣的 Sharpe（DSR 思路的可算版本）。

    多重检验下，N 次独立尝试里最大 Sharpe 的期望本身就 > 0。
    这里给出该期望值作为「零假设下你本来就能挑到多少」的基准。
    """
    if not np.isfinite(sharpe) or n_obs < 3 or n_trials < 1:
        return dict(se=np.nan, expected_max=np.nan, adjusted=np.nan)

    # ★ 单位必须自始至终一致，否则 t 会被系统性夸大。
    # 第一版把【周期】口径的方差项和【年化】口径的 Sharpe 混在一个式子里，
    # 算出 se=0.09、t=3.43，而同一条序列的 NW lag=0 t 只有 1.02 ——
    # 夸大 3.4 倍。两个量测的是同一件事，对不上就说明有一个错了。
    # 正确做法：先在周期口径算 SE，再乘 sqrt(ppy) 换成年化。
    sr_p = sharpe / np.sqrt(ppy)                      # 周期口径 Sharpe
    se_p = np.sqrt((1.0 + 0.5 * sr_p ** 2) / n_obs)   # 周期口径 SE
    se = se_p * np.sqrt(ppy)                          # 年化口径 SE
    if n_trials == 1:
        exp_max = 0.0
    else:
        # E[max of N standard normals] 的经典近似
        g = 0.5772156649
        e = ((1 - g) * _z(1 - 1.0 / n_trials)
             + g * _z(1 - 1.0 / (n_trials * np.e)))
        exp_max = e * se
    return dict(se=se, expected_max=exp_max, adjusted=sharpe - exp_max)


def _z(p: float) -> float:
    """标准正态分位数（Acklam 近似，避免依赖 scipy）。"""
    p = min(max(p, 1e-12), 1 - 1e-12)
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    pl = 0.02425
    if p < pl:
        q = np.sqrt(-2 * np.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q
                + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > 1 - pl:
        q = np.sqrt(-2 * np.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q
                 + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r
            + a[5]) * q / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r
                            + b[4]) * r + 1)


def check_deflated_sharpe(rets: pd.Series, ppy: float, rep: AuditReport,
                          n_trials: int = 1) -> None:
    """③ 多重检验折扣。

    ★ n_trials 必须由【用户】提供 —— 只有他知道自己试过多少配置。
    默认 1 并明确说明「这等于假设你一次就成」，因为静默假设 1
    等于替用户隐瞒多重检验。
    """
    r = pd.Series(rets).dropna().astype(float)
    if len(r) < 12:
        rep.skip("多重检验折扣", "序列不足 12 期")
        return

    a = annualize(r, ppy)
    sr = a["sharpe"]
    if not np.isfinite(sr):
        rep.skip("多重检验折扣", "Sharpe 无法计算（波动为 0）")
        return

    res = deflated_sharpe(sr, len(r), n_trials, ppy)
    rep.stats["sharpe"] = sr
    rep.stats["sharpe_se"] = res["se"]
    rep.stats["sharpe_trials"] = n_trials
    rep.stats["sharpe_adjusted"] = res["adjusted"]

    t_sr = sr / res["se"] if res["se"] and np.isfinite(res["se"]) else np.nan
    base = (f"Sharpe {sr:.2f}，标准误 {res['se']:.2f}"
            f"（{len(r)} 期，t={t_sr:.2f}）")

    if n_trials <= 1:
        rep.add(WARN, "多重检验折扣未计入",
                base + "\n按【只试过 1 个配置】计算 —— 这是默认假设，不是事实",
                "若你试过 N 个参数/因子/股票池组合才选出这一个，"
                "上面的 Sharpe 必须折扣。请用 n_trials 传入真实尝试次数："
                "试过 20 个配置时，零假设下最大 Sharpe 的期望就有 "
                f"{deflated_sharpe(sr, len(r), 20, ppy)['expected_max']:.2f}",
                section=SECTION)
        return

    exp_max = res["expected_max"]
    adj = res["adjusted"]
    detail = (base + f"\n试过 {n_trials} 个配置 ⇒ 零假设下最大 Sharpe 期望 "
              f"{exp_max:.2f}，折扣后 {adj:.2f}")
    if adj <= 0:
        rep.add(BLOCK, "折扣后 Sharpe 不为正", detail,
                f"试 {n_trials} 次挑出 Sharpe {sr:.2f}，"
                "与随机挑选无法区分。这不代表策略一定无效，"
                "但这份样本不能作为证据 —— 需要样本外验证",
                section=SECTION)
        return
    if adj < 0.5 * sr:
        rep.add(WARN, "折扣后 Sharpe 大幅下降", detail,
                f"折扣掉了 {(1 - adj / sr):.0%}。汇报时应给折扣后的数字",
                section=SECTION)
        return
    rep.add(OK, "多重检验折扣", detail + " —— 折扣后仍稳健", section=SECTION)


def check_drawdown(rets: pd.Series, rep: AuditReport) -> None:
    """④ 回撤与单期极值贡献。

    ★ 单期贡献过大是「一期定生死」，与年份集中度同源但更极端：
    那一期若是数据错误（如未处理的除权），整条净值就是假的。
    """
    r = pd.Series(rets).dropna().astype(float)
    if len(r) < 6:
        rep.skip("回撤与极值", "序列不足 6 期")
        return

    nav = (1 + r).cumprod()
    peak = nav.cummax()
    dd = nav / peak - 1.0
    mdd = float(dd.min())
    rep.stats["max_drawdown"] = mdd

    total = float(nav.iloc[-1] - 1.0)
    best = float(r.max())
    best_at = r.idxmax()
    share = abs(best / total) if abs(total) > 1e-9 else np.inf
    rep.stats["best_period_share"] = share

    # ★ 「单期占累计的比例」这个量在累计收益接近 0 时会爆掉（分母趋零），
    # 所以它【不能】用固定门槛判。实测纯噪声下 p90 就有 0.78、p95 达 1.45,
    # 用 0.30 当门槛误报率 27%。
    # 改判：看这一期在【收益分布】里有多极端（是不是离群点），
    # 那才是「数据可能有错」的真信号，且与累计收益大小无关。
    sd = float(r.std(ddof=1))
    z_best = (best - float(r.mean())) / sd if sd > 0 else np.nan
    rep.stats["best_period_z"] = z_best

    parts = [f"最大回撤 {mdd:.2%}"]
    trough = dd.idxmin()
    after = dd[dd.index > trough]
    rec = after[after >= -1e-9]
    parts.append(f"低点 {trough.date() if hasattr(trough, 'date') else trough}"
                 + (f"，{len(after[after.index <= rec.index[0]])} 期后收复"
                    if len(rec) else "，样本末仍未收复"))
    detail = "；".join(parts)

    at = best_at.date() if hasattr(best_at, "date") else best_at
    extra = (f"\n最好的一期（{at}）收益 {best:+.2%}"
             f"（距均值 {z_best:.1f} 个标准差）")
    if np.isfinite(share):
        extra += f"，占累计 {total:+.2%} 的 {share:.0%}"

    if np.isfinite(z_best) and z_best > SINGLE_PERIOD_Z:
        rep.add(WARN, "单期收益是离群点",
                detail + extra,
                f"这一期偏离均值 {z_best:.1f} 个标准差，"
                f"在 {len(r)} 期样本里属于异常值。"
                "若它是数据问题（未处理的除权、错误的复权因子、"
                "停牌后异常跳空），整条净值就是假的 —— "
                "请单独核对那一期的持仓与价格。"
                "★ 本项只说它异常，不说它一定错",
                section=SECTION)
        return
    rep.add(OK, "回撤与极值", detail + extra, section=SECTION)
