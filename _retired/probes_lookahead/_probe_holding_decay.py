"""持有期第 1 日收益集中度探针（已否决，禁止接入正式检查）。

这是一个研究工具，不是 strategy-audit 的 CHECKS 项。它测量每个调仓区间
第 1 个持有日相对后续持有日的异常程度：

    z = (mean(day_1) - mean(day_2..K)) / sd(day_2..K)

初步实验中，干净的月频反转/动量组合 |z| 已约 2.4，而故意泄漏一天的版本
为 4.85/8.69；并且泄漏方向随因子族改变。因此「z 显著」不是可信的前视
判据。此文件保留构造与否决理由，避免未来把未标定的直觉包装成报警。

允许重新考虑登记的唯一条件（缺一项即继续否决）：
  * >= 3 因子族、>= 2 独立价格面板、月频和周频各一组；
  * 干净 |z| 的 95 分位 < 泄漏一天 |z| 的 5 分位；分布重叠即否决；
  * 门槛预注册为干净分布的 99 分位，且负样本包含月初效应强的样本期；
  * 即使通过，正式项最多 WARN，措辞只能要求核对因子时间戳，不能断言前视。
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from strategy_audit.core import align, daily_path  # noqa: E402


def holding_day_profile(weights: pd.DataFrame, prices: pd.DataFrame,
                        max_day: int = 20) -> pd.DataFrame:
    """返回每段持有期中第 k 天的平均组合收益和探索性 z。

    ``weights`` 与 ``prices`` 必须已经是 date × code 矩阵。各段长度不同，
    只在该日存在时纳入均值；这正是为何其结果不能直接当成正式检验。
    """
    wm, pm = align(weights, prices)
    rows = []
    for start, end in zip(wm.index[:-1], wm.index[1:]):
        segment = daily_path(wm.loc[[start, end]], pm)
        for day, ret in enumerate(segment.iloc[:max_day], start=1):
            rows.append((start, day, float(ret)))
    raw = pd.DataFrame(rows, columns=["rebalance", "holding_day", "ret"])
    if raw.empty:
        return pd.DataFrame(columns=["mean_ret", "n", "z_day1"])
    prof = raw.groupby("holding_day")["ret"].agg(mean_ret="mean", n="size")
    later = prof.loc[prof.index >= 2, "mean_ret"]
    if len(later) >= 2 and float(later.std(ddof=1)) > 0 and 1 in prof.index:
        prof["z_day1"] = (prof.loc[1, "mean_ret"] - float(later.mean())) / \
            float(later.std(ddof=1))
    else:
        prof["z_day1"] = np.nan
    return prof


if __name__ == "__main__":
    raise SystemExit(
        "这是已否决的研究探针；请先满足模块注释中的完整标定要求，"
        "不得把输出当作 strategy-audit 的前视结论。")
