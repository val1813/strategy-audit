"""命令行入口。只是 API 的薄封装 —— 所有逻辑在 _api.py。

    strategy-audit --weights w.csv --prices panel.parquet
    strategy-audit --weights w.csv --prices panel.parquet --net net.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ._api import audit_strategy


def _read(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"文件不存在：{p}")
    if p.suffix.lower() in (".parquet", ".pq"):
        return pd.read_parquet(p)
    return pd.read_csv(p)


def _read_series(path: str, what: str) -> pd.Series:
    """读单列时间序列：date + 一列数值。"""
    d = _read(path)
    if "date" not in d.columns:
        raise SystemExit(f"{what} 需要 date 列，实际列：{list(d.columns)}")
    vals = [c for c in d.columns if c != "date"]
    if len(vals) != 1:
        raise SystemExit(
            f"{what} 需要恰好一列数值（除 date 外），实际：{vals}")
    s = pd.Series(pd.to_numeric(d[vals[0]], errors="coerce").values,
                  index=pd.to_datetime(d["date"]))
    return s.sort_index()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="strategy-audit",
        description="策略审计器 —— 查你的回测到净值这一段有没有偷分（数据由你提供）")
    ap.add_argument("--weights", required=True,
                    help="权重面板（parquet/csv）：date,code,weight")
    ap.add_argument("--prices", required=True,
                    help="价格面板（parquet/csv）：date,code,close")
    ap.add_argument("--net", default=None,
                    help="可选：你自己算的【净】收益 date,<值>，用于毛净对账")
    ap.add_argument("--benchmark", default=None,
                    help="可选：基准收益 date,<值>，盈亏平衡成本按超额算")
    ap.add_argument("--name", default="策略审计", help="报告标题")
    args = ap.parse_args(argv)

    rep = audit_strategy(
        _read(args.weights),
        _read(args.prices),
        net_returns=_read_series(args.net, "--net") if args.net else None,
        benchmark=(_read_series(args.benchmark, "--benchmark")
                   if args.benchmark else None),
        name=args.name,
    )
    print(rep.text())

    # ★ 退出码：有 BLOCK 返回 1，方便进 CI。
    # 但【不】把 WARN 也算失败 —— WARN 是「打折使用」，
    # 让 WARN 挂 CI 会逼客户去关告警。
    return 1 if rep.blockers else 0


if __name__ == "__main__":
    raise SystemExit(main())
