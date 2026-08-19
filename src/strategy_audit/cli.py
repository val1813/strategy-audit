"""命令行入口。给文件就行，不用记参数名。

    strategy-audit 持仓.csv 行情.parquet
    strategy-audit 净值.csv
    strategy-audit --demo                    # 没有数据？先看看能干什么

★ 位置参数而非 --weights/--prices：
用户不该为了用工具去查「哪个表该配哪个 flag」。识别交给 detect.py，
识别结果会打在报告里让他核对。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ._api import audit

DEMO_NOTE = """
★ 这是内置合成数据的演示（不是你的数据）。
  面板：30 只标的 / 3 年日频 / 其中 6 只中途退市（退市前跌 80%）
  策略：月频等权 8 只
  用它先看清报告长什么样、能审哪些项，再换成你自己的数据。
"""


def _demo_inputs():
    """造一份带已知缺陷的合成数据，零准备也能跑通。"""
    rng = np.random.default_rng(3)
    dates = pd.bdate_range("2021-01-04", "2023-12-29")
    codes = [f"{i:06d}.SZ" for i in range(1, 31)]
    px = pd.DataFrame(
        {c: 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.018, len(dates))))
         for c in codes}, index=dates)
    # 6 只中途退市，退市前 20 日跌 80%
    for i in range(6):
        cut = int(len(dates) * (0.4 + 0.4 * (i + 1) / 6))
        k = min(20, cut)
        decay = (1.0 - 0.8) ** (np.arange(1, k + 1) / k)
        px.iloc[cut - k:cut, i] = px.iloc[cut - k:cut, i].values * decay
        px.iloc[cut:, i] = np.nan

    reb = []
    for d in px.resample("ME").last().index:
        j = px.index.get_indexer([d], method="ffill")[0]
        if j >= 0:
            reb.append(px.index[j])
    reb = sorted(set(reb))

    rows = []
    for t in reb:
        pool = [c for c in px.columns if np.isfinite(px.loc[t, c])]
        if len(pool) < 8:
            continue
        for c in rng.choice(pool, 8, replace=False):
            rows.append((t, c, 1 / 8))
    w = pd.DataFrame(rows, columns=["date", "code", "weight"])
    p = px.stack().rename("close").reset_index()
    p.columns = ["date", "code", "close"]
    return w, p


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="strategy-audit",
        description="策略审计器 —— 查你的回测到净值这一段有没有偷分。"
                    "给什么审什么：持仓表、行情表、净值曲线，任意组合。",
        epilog="例：strategy-audit 持仓.csv 行情.parquet\n"
               "    strategy-audit 净值.csv --trials 20\n"
               "    strategy-audit --demo",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="*",
                    help="任意个数据文件（csv/parquet/xlsx）。"
                         "权重表/价格表/净值曲线自动识别，顺序无关")
    ap.add_argument("--demo", action="store_true",
                    help="用内置合成数据演示（不需要你的数据）")
    ap.add_argument("--trials", type=int, default=1, metavar="N",
                    help="你试过多少个配置才选出这一个（用于多重检验折扣）")
    ap.add_argument("--net", metavar="FILE",
                    help="净收益/净值文件（用于毛净对账；该角色不能自动猜）")
    ap.add_argument("--benchmark", metavar="FILE",
                    help="基准收益/净值文件（盈亏平衡按超额收益计算）")
    ap.add_argument("--name", default="策略审计", help="报告标题")
    ap.add_argument("--quiet-detection", action="store_true",
                    help="不打印输入识别明细（不推荐：识别错了你会看不出来）")
    args = ap.parse_args(argv)

    if args.demo:
        print(DEMO_NOTE)
        w, p = _demo_inputs()
        rep = audit(w, p, n_trials=args.trials, name="演示（内置合成数据）",
                    show_detection=not args.quiet_detection)
        print(rep.text())
        # ★ 退出码规则对 --demo 也必须一致。
        # 演示数据【故意】含缺陷（6 只退市股），报告里有 BLOCK 却返回 0
        # 会让 CI 学到「有 BLOCK 也算过」——
        # 同一个工具在不同入口给不同语义，是最难查的那种坑。
        return 1 if rep.blockers else 0

    if not args.files:
        ap.print_help()
        print("\n★ 没给文件。想先看看报告长什么样，跑：strategy-audit --demo")
        return 2

    for f in args.files:
        if not Path(f).exists():
            raise SystemExit(f"文件不存在：{f}")

    rep = audit(*args.files, net=args.net, benchmark=args.benchmark,
                n_trials=args.trials, name=args.name,
                show_detection=not args.quiet_detection)
    print(rep.text())

    # ★ 退出码：有 BLOCK 返回 1，方便进 CI。
    # WARN 不算失败 —— 让 WARN 挂 CI 会逼用户去关告警。
    return 1 if rep.blockers else 0


if __name__ == "__main__":
    raise SystemExit(main())
