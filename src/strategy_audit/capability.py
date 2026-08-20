"""能力矩阵：你给了什么 ⇒ 能审什么 ⇒ 缺的补上能多审什么。

★ 为什么这一层比任何单个检查都重要
--------------------------------
用户来的时候不知道自己该给什么，也不知道给少了会漏掉什么。
第一版缺列直接 BLOCK，等于把「我们的格式要求」变成他的工作量 ——
而他本来就是因为不确定回测有没有问题才来的。

现在反过来：**能审的先审，不能审的明确说出来并给出补什么**。
报告顶部先打这张表，用户一眼看到「我给了净值 ⇒ 4 项能审、8 项要权重」。

★ 「不能审」必须和「审过且通过」严格区分
------------------------------------
这是 factor-audit 的核心教训（静默跳过 = 客户以为查过了）。
能力矩阵把这件事前置：不是等到报告末尾才说「未能检查」，
而是一开始就把边界画清楚。
"""

from __future__ import annotations

from dataclasses import dataclass

# 输入种类
W = "weights"       # 权重面板
P = "prices"        # 价格面板
NAV = "nav"         # 净值或收益率序列（可以是权重+价格【重算】出来的）
NET = "net"         # 净收益序列（用于毛净对账）
BENCH = "bench"     # 基准序列
AMT = "amount"      # 价格面板里的成交额列（族五容量用）
OWN_NAV = "own_nav"  # 【客户自己汇报的】那条净值曲线
OPEN = "open"
FLAGS = "flags"
SIG = "signals"
SIG_ALT = "signals_alt"
DELISTED = "delisted"
PARAMS = "params"
TRADES = "trades"

# ★ 为什么 OWN_NAV 必须与 NAV 分开
# --------------------------------
# NAV 在权重+价格齐全时是【工具自己重算】出来的，所以它总是「有」。
# 而「自报净值对账」需要的是客户【亲手交上来的】那一条 —— 两者
# 不区分的话，能力矩阵会把对账项显示成「能审」，而它根本没得对。
# 那正是本工具最忌讳的事：把「没查」显示成「查过了」。

_LABEL = {
    W: "权重面板（date/code/weight）",
    P: "价格面板（date/code/close）",
    NAV: "净值或收益率序列",
    NET: "净收益序列",
    BENCH: "基准序列",
    AMT: "成交额列（价格表里加一列 amount/成交额）",
    OWN_NAV: "你自己汇报的净值曲线",
    OPEN: "开盘价列（价格表里加 open/开盘价）",
    FLAGS: "交易标志位（涨跌停价/停牌/ST）",
    SIG: "信号强度面板（date/code/score）",
    SIG_ALT: "第二家供应商信号面板",
    DELISTED: "权威退市名单",
    PARAMS: "显式声明的策略参数",
    TRADES: "交易明细（entry_date/code[/exit_date]）",
}


@dataclass(frozen=True)
class Check:
    key: str
    name: str
    section: str
    needs: tuple           # 必需输入
    catches: str           # 一句话说它抓什么


# ★ 每项检查声明自己需要什么。新增检查必须在这里登记，
# 否则能力矩阵会漏报 —— 这是防止「悄悄少查一项」的登记表。
CHECKS = (
    # 族七：净值质量。★ 排在族三之前 —— 族三审「这个数字可不可信」，
    # 而它默认曲线本身是真的；族七审的正是那个前提。
    Check("smoothing", "平滑与滞后定价", "净值质量", (NAV,),
          "收益自相关 ⇒ 波动被低估、Sharpe 被高估"),
    Check("stale_nav", "停滞估值", "净值质量", (NAV,),
          "净值原地不动的占比 ⇒ 估值没更新"),
    Check("dressing", "期末修饰", "净值质量", (NAV,),
          "期末最后一日收益是否系统性偏高"),

    # 族三：只要一条曲线就能跑（门槛最低，所以排第一）
    Check("year_conc", "年份集中度", "策略层显著性", (NAV,),
          "超额是否集中在极少数年份"),
    Check("nw_lag", "NW lag 敏感性", "策略层显著性", (NAV,),
          "t 值是否靠 lag 选择撑起来"),
    Check("dsr", "多重检验折扣", "策略层显著性", (NAV,),
          "按你试过的配置数折扣后 Sharpe 还剩多少"),
    Check("drawdown", "回撤与恢复", "策略层显著性", (NAV,),
          "最大回撤、恢复期、单期极值贡献"),

    # 族二：前视与记账
    Check("day0", "调仓日对齐", "前视与记账", (W, P),
          "用 t 日收盘定权重却吃了 t 日涨幅"),
    Check("w_look", "权重前视", "前视与记账", (W, P),
          "权重与同期收益相关（vw 用期末市值）"),
    Check("univ", "股票池生存者", "前视与记账", (W, P),
          "持仓只落在活到末日的标的上"),
    Check("member", "缺价记账（停牌/退市）", "前视与记账", (W, P),
          "停牌被当退市、三种政策的净值区间"),

    # 输入契约：★ 自报净值对账排在所有检查之前 ——
    # 若你汇报的净值和权重表算出来的不是一条曲线，那下面审的就不是你的策略。
    Check("nav_recon", "自报净值对账", "输入契约", (W, P, OWN_NAV),
          "你汇报的净值与按权重+价格重算的是不是同一条曲线"),

    # 族一：换手与成本
    Check("to_basis", "换手口径", "换手与成本", (W, P),
          "朴素 vs 漂移调整口径"),
    Check("to_implied", "反推换手偏差", "换手与成本", (W, P),
          "自相关反推与实际权重变化的比值"),
    Check("breakeven", "盈亏平衡成本", "换手与成本", (W, P),
          "多大的单边成本吃掉全部超额"),
    Check("reconcile", "毛净对账", "换手与成本", (W, P, NET),
          "你声称扣的成本，实际扣了多少 bp"),

    # 族四：风险身份（不问净值对不对，问风险预算编得对不对）
    Check("breadth", "残差有效注数", "风险身份", (W, P),
          "报的持仓数背后其实只有几注独立的赌"),
    Check("breadth_ctrl", "同规模对照", "风险身份", (W, P),
          "注数压缩是你选股的性质，还是同规模组合都这样"),
    Check("breadth_enb", "与 ENB 并列", "风险身份", (W, P),
          "和 Meucci 有效注数是不是同一个量"),

    # 族五：容量与可成交性。★ 除「不可成交权重」外都要成交额列。
    Check("untradable", "不可成交权重", "容量与可成交性", (W, P),
          "调仓日涨跌停/停牌的权重占多少"),
    Check("capacity", "资金容量上限", "容量与可成交性", (W, P, AMT),
          "按成交额反解这份持仓能管多少钱"),
    Check("liq_tilt", "流动性集中度", "容量与可成交性", (W, P, AMT),
          "超额是否来自最难成交的标的"),
    Check("size_decay", "规模衰减曲线", "容量与可成交性", (W, P, AMT, NAV),
          "规模放大到哪一档超额归零"),

    # 族六：处方层。★ 唯一会说「怎么改」的一族，每项自带过拟合闸门。
    Check("to_value", "换手的边际价值", "处方（怎么改）", (W, P),
          "跟什么都不做比，这次调仓在毛口径上买到了什么"),
    Check("to_split", "换手成分拆分", "处方（怎么改）", (W, P),
          "换手里多少是名单换血、多少是权重微调"),
    Check("prescribe", "削换手处方", "处方（怎么改）", (W, P),
          "有没有一个无需预测的改动能确定改善净收益"),

    # 参数邻域只体检声明档位，不寻优、不推荐换档。
    Check("holding_neighborhood", "持有期邻域体检", "参数邻域体检", (W, P, PARAMS),
          "声明持有期是不是孤立尖峰，还是稳定平台"),
    Check("topn_neighborhood", "Top-N 邻域体检", "参数邻域体检", (W, P, SIG, PARAMS),
          "用几何年化检查极窄选股口径的宽度梯度"),
    Check("entry_neighborhood", "入场时点体检", "参数邻域体检", (W, P, OPEN),
          "open 与 close 入场的配对差异"),
    Check("signal_consistency", "信号-持仓自洽", "参数邻域体检", (W, SIG),
          "信号排序能否重现实际买入，抓排序方向与 tie bug"),
    Check("vendor_pair", "跨供应商配对", "参数邻域体检", (P, SIG, SIG_ALT, PARAMS),
          "alpha 是策略的还是供应商口径的"),
    Check("deferred_exit", "跌停出场顺延", "参数邻域体检", (P, FLAGS, TRADES),
          "出场撞在跌停封板日、顺延后差多少（判板用行情价不用成交价）"),
)


def label(kind: str) -> str:
    """输入种类的人类可读名字。★ 报告里不许出现内部键名。"""
    return _LABEL.get(kind, kind)


def available(have: set) -> tuple[list, list]:
    """按已有输入切分：能跑的 / 跑不了的。"""
    ok = [c for c in CHECKS if set(c.needs) <= have]
    no = [c for c in CHECKS if not set(c.needs) <= have]
    return ok, no


def can_run(key: str, have: set) -> bool:
    """某项检查在当前输入下能否运行。

    这是执行调度与能力矩阵共用的唯一输入判据。运行时仍可能因样本过少、
    缺失值等数据质量原因跳过，但绝不能出现「矩阵说缺输入，正文却运行」。
    """
    for check in CHECKS:
        if check.key == key:
            return set(check.needs) <= set(have)
    raise KeyError(f"未知检查键：{key}")


def missing_value(have: set) -> list:
    """算出「补哪个输入能多解锁几项」，按收益排序。

    ★ 这是给用户的行动建议，必须按【解锁项数】排序而不是按我们
    觉得哪个重要 —— 他要的是投入产出比。
    """
    ok, no = available(have)
    gains = []
    # ★ 单个输入 + 常见【组合】都要算。
    # 第一版只试单个，于是「只有净值」的用户看到的是
    # 「补权重面板 ⇒ 多 1 项」—— 而权重+价格一起补能多 8 项。
    # 用户看到 1 项就不会去补了，等于把最有价值的建议藏了起来。
    kinds_all = (W, P, NAV, NET, BENCH, AMT, OWN_NAV, OPEN, FLAGS,
                 SIG, SIG_ALT, DELISTED, PARAMS, TRADES)
    combos = [(k,) for k in kinds_all]
    combos += [(W, P), (W, P, AMT), (W, P, NET), (W, P, AMT, NET),
               (W, P, OWN_NAV), (W, P, AMT, NET, OWN_NAV)]
    seen = set()
    for kinds in combos:
        need = tuple(k for k in kinds if k not in have)
        if not need or need in seen:
            continue
        seen.add(need)
        more, _ = available(have | set(need))
        delta = len(more) - len(ok)
        if delta > 0:
            gains.append((need, delta, [c for c in more if c not in ok]))
    # 先按解锁项数降序，同项数时优先要求少的组合
    gains.sort(key=lambda x: (-x[1], len(x[0])))
    # 去掉被更省输入的方案完全支配的建议
    out = []
    for need, delta, which in gains:
        dominated = any(set(n2) < set(need) and d2 >= delta
                        for n2, d2, _ in gains)
        if not dominated:
            out.append((need, delta, which))
    return out


def matrix_text(have: set) -> str:
    """报告顶部那张能力表。"""
    ok, no = available(have)
    lines = []
    lines.append("  你给了：" + ("、".join(
        _LABEL[k] for k in (W, P, NAV, NET, BENCH, AMT, OWN_NAV, OPEN, FLAGS,
                            SIG, SIG_ALT, DELISTED, PARAMS, TRADES) if k in have)
        or "（无法识别的输入）"))
    lines.append("")
    lines.append(f"  能审 {len(ok)}/{len(CHECKS)} 项：")
    cur = ""
    for c in ok:
        if c.section != cur:
            cur = c.section
            lines.append(f"    【{cur}】")
        lines.append(f"      ✓ {c.name} —— {c.catches}")

    if no:
        lines.append("")
        lines.append(f"  审不了 {len(no)} 项（缺输入，【不是】查过通过）：")
        for c in no:
            lack = "、".join(_LABEL[k].split("（")[0]
                             for k in c.needs if k not in have)
            lines.append(f"      ✗ {c.name} —— 需要 {lack}")

        gains = missing_value(have)
        if gains:
            lines.append("")
            lines.append("  补什么能多审几项（按性价比排序）：")
            for need, delta, which in gains[:4]:
                what = " + ".join(_LABEL[k] for k in need)
                names = "、".join(c.name for c in which)
                if len(names) > 46:
                    names = names[:44] + "…"
                lines.append(f"      + {what}")
                lines.append(f"        ⇒ 多审 {delta} 项（{names}）")
    return "\n".join(lines)
