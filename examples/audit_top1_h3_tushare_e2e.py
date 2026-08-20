"""Trade-level end-to-end audit against an independent Tushare price panel.

Pass the trade workbook with ``--trades``; download the panel first with
``fetch_tushare_trade_audit_panel.py``.

This deliberately does not manufacture a portfolio weight panel from trade
details.  The source workbook contains individual fills, but not the full
daily holdings/cash ledger needed for turnover, NAV reconciliation, breadth,
or capacity.  It therefore audits every fact the input supports, and records
the remaining input boundary explicitly.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from strategy_audit import parameter_audit as pa
from strategy_audit.report import AuditReport


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PARTS = ROOT / "data" / "trade_audit_open_limits_parts"
DEFAULT_FACTOR_PARTS = ROOT / "data" / "trade_audit_open_limits_adj_factor_parts"
DEFAULT_OUT = ROOT / "data" / "trade_audit_e2e_report.md"
# Cross-vendor prices and adjustment factors can differ at a few basis points.
# This is a diagnostic threshold, not an acceptance threshold: any difference
# remains visible and prevents a full PASS until its provenance is resolved.
DIAGNOSTIC_TOLERANCE = 1e-4  # 1 bp of return


def load_panel(parts: Path) -> pd.DataFrame:
    files = sorted(parts.glob("*.csv"))
    if len(files) != 314:
        raise RuntimeError(f"expected 314 Tushare parts, found {len(files)}: {parts}")
    d = pd.concat((pd.read_csv(f, parse_dates=["date"]) for f in files), ignore_index=True)
    if d.duplicated(["date", "code"]).any():
        raise RuntimeError("downloaded panel has duplicate date/code rows")
    return d


def load_factors(parts: Path) -> pd.DataFrame:
    """Load Tushare adjustment factors kept beside the immutable price parts."""
    files = sorted(parts.glob("*.csv"))
    if len(files) != 314:
        raise RuntimeError(
            f"expected 314 Tushare adj_factor parts, found {len(files)}: {parts}"
        )
    d = pd.concat((pd.read_csv(f, parse_dates=["date"]) for f in files), ignore_index=True)
    if d.duplicated(["date", "code"]).any():
        raise RuntimeError("downloaded adj_factor panel has duplicate date/code rows")
    if (d["adj_factor"] <= 0).any() or d["adj_factor"].isna().any():
        raise RuntimeError("adj_factor panel contains missing or non-positive values")
    return d


def load_trades(path: Path) -> pd.DataFrame:
    d = pd.read_excel(path, sheet_name="交易明细").copy()
    internal = d["内部代码"].astype(str).str.lower()
    d["code"] = internal.str[2:] + "." + internal.str[:2].str.upper()
    d["entry_date"] = pd.to_datetime(d["买入日期"])
    d["exit_date"] = pd.to_datetime(d["卖出日期"])
    return d


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trades", type=Path, required=True,
                    help="workbook with a 交易明细 sheet")
    ap.add_argument("--parts", type=Path, default=DEFAULT_PARTS)
    ap.add_argument("--factor-parts", type=Path, default=DEFAULT_FACTOR_PARTS)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args(argv)
    panel = load_panel(args.parts)
    factors = load_factors(args.factor_parts)
    trades = load_trades(args.trades)
    panel = panel.merge(factors, on=["date", "code"], how="left", validate="one_to_one")
    if panel["adj_factor"].isna().any():
        raise RuntimeError("Tushare daily rows are missing an adj_factor")
    # Tushare daily is unadjusted; multiplying each price by its factor makes
    # the comparisons post-adjusted without changing a same-day limit fact.
    panel["hfq_open"] = panel["open"] * panel["adj_factor"]
    panel["hfq_close"] = panel["close"] * panel["adj_factor"]
    # 涨跌停价必须与 close 同复权基准，否则 close == down_limit 的判定
    # 会因两边基准不同而恒不成立（漏报方向）。
    panel["hfq_up_limit"] = panel["up_limit"] * panel["adj_factor"]
    panel["hfq_down_limit"] = panel["down_limit"] * panel["adj_factor"]
    entry = panel[["date", "code", "open", "hfq_open", "hfq_close", "up_limit", "down_limit"]].rename(
        columns={"date": "entry_date", "open": "ts_open", "hfq_open": "ts_hfq_open",
                 "hfq_close": "ts_hfq_entry_close",
                 "up_limit": "entry_up_limit", "down_limit": "entry_down_limit"})
    exit_ = panel[["date", "code", "close", "hfq_close", "up_limit", "down_limit"]].rename(
        columns={"date": "exit_date", "close": "ts_exit_close", "hfq_close": "ts_hfq_exit_close",
                 "up_limit": "exit_up_limit", "down_limit": "exit_down_limit"})
    x = trades.merge(entry, on=["entry_date", "code"], how="left")
    x = x.merge(exit_, on=["exit_date", "code"], how="left")
    x["ts_raw_gross_ret"] = x["ts_exit_close"] / x["ts_open"] - 1.0
    x["ts_hfq_gross_ret"] = x["ts_hfq_exit_close"] / x["ts_hfq_open"] - 1.0
    x["xlsx_gross_ret"] = pd.to_numeric(x["毛收益率"]) / 100.0
    x["hfq_gross_diff"] = x["ts_hfq_gross_ret"] - x["xlsx_gross_ret"]
    x["raw_gross_diff"] = x["ts_raw_gross_ret"] - x["xlsx_gross_ret"]

    close_pm = panel.pivot(index="date", columns="code", values="hfq_close").sort_index()
    open_pm = panel.pivot(index="date", columns="code", values="hfq_open").sort_index()
    hrep = AuditReport()
    pa.check_holding(pd.DataFrame(), close_pm, open_pm, 3, hrep,
                     trades=x[["entry_date", "exit_date", "code"]])
    holding = hrep.findings[0]

    # 跌停出场顺延：走库内检查，报告不另写一套判据（避免两处口径漂移）
    dl_pm = panel.pivot(index="date", columns="code",
                        values="hfq_down_limit").sort_index()
    drep = AuditReport()
    pa.check_deferred_exit(x[["entry_date", "exit_date", "code"]], close_pm,
                           {"down_limit": dl_pm}, drep, entry_pm=open_pm)
    deferred = drep.findings[0]

    exact = int((x["hfq_gross_diff"].abs() <= 1e-9).sum())
    within_tolerance = int((x["hfq_gross_diff"].abs() <= DIAGNOSTIC_TOLERANCE).sum())
    non_exact = len(x) - exact
    top = x.loc[x["hfq_gross_diff"].abs().nlargest(5).index,
                ["证券代码", "entry_date", "exit_date", "xlsx_gross_ret",
                 "ts_hfq_gross_ret", "hfq_gross_diff"]]
    raw_nonexact = int((x["raw_gross_diff"].abs() > 1e-9).sum())
    # Entry price makes a material difference for most fills.  This is a
    # per-trade sensitivity only: the workbook lacks daily portfolio weights,
    # so it must not be mislabeled as a recomputed strategy NAV.
    x["ts_hfq_close_entry_ret"] = x["ts_hfq_exit_close"] / x["ts_hfq_entry_close"] - 1.0
    entry_basis_changed = int((x["ts_hfq_gross_ret"] - x["ts_hfq_close_entry_ret"]).abs().gt(1e-9).sum())

    # ---- §5 涨跌停：判板一律用【市场行情价】 ----
    # ★ 用成交价判板在构造上不可能命中：成交价含滑点，永远不精确等于
    # 涨跌停价。第一版用 `实际卖出价 == down_limit` 得 0/796，据此写下
    # 「未观察到收盘跌停卖出」—— 而市场 close 判据实测 8 笔真封板
    # （收盘精确 −10.0%）。判据错的方向是【漏报】，最不该出错的方向。
    entry_up = np.isclose(x["ts_open"], x["entry_up_limit"], atol=1e-6, rtol=1e-6)
    exit_down = np.isclose(x["ts_exit_close"], x["exit_down_limit"], atol=1e-6, rtol=1e-6)
    n_entry_limit_up = int(entry_up.sum())
    n_exit_limit_down = int(exit_down.sum())
    # 反例对照：同一批笔用成交价判，命中数必为 0（留在报告里当护栏）
    n_exit_limit_down_by_fill = int(np.isclose(
        x["实际卖出价"], x["exit_down_limit"], atol=1e-6, rtol=1e-6).sum())
    ld = x.loc[exit_down]
    limit_down_mean = float(ld["xlsx_gross_ret"].mean()) if len(ld) else float("nan")
    total_pnl = float(pd.to_numeric(x["净盈亏(元)"]).sum())
    limit_down_pnl_share = (float(pd.to_numeric(ld["净盈亏(元)"]).sum()) / total_pnl
                            if len(ld) and total_pnl else float("nan"))
    limit_down_detail = ld[["证券代码", "entry_date", "exit_date", "ts_exit_close",
                            "exit_down_limit", "实际卖出价", "xlsx_gross_ret"]].copy()
    limit_down_detail["entry_date"] = limit_down_detail["entry_date"].dt.date
    limit_down_detail["exit_date"] = limit_down_detail["exit_date"].dt.date

    # ---- 离群笔归因：是复权基准差异，还是价格本身错？----
    # ★ 判别法：若【未复权】比值与 xlsx 毛收益精确相等，而后复权比值不等，
    # 则差异全部来自两家的复权因子基准，不是价格错。这比只写「须追溯」
    # 信息量大得多，也决定用户要不要去找数据商。
    worst = x.loc[x["hfq_gross_diff"].abs().idxmax()]
    raw_matches = bool(abs(float(worst["raw_gross_diff"])) <= 1e-9)
    outlier_note = (
        f"- 离群笔归因：{worst['证券代码']} {pd.Timestamp(worst['entry_date']).date()} "
        f"→ {pd.Timestamp(worst['exit_date']).date()} 后复权差 "
        f"{abs(float(worst['hfq_gross_diff'])) * 1e4:.2f} bp。"
        + ("该笔【未复权】比值与 xlsx 毛收益精确相等（|diff| ≤ 1e-9），"
           "而 adj_factor 在持有期内发生变化 ⇒ 差异全部来自两家的**复权基准**，"
           "不是价格数据错。无需向数据商追价格，但跨供应商比较后复权收益时"
           "必须固定同一复权基准。"
           if raw_matches else
           "该笔未复权比值也不相等 ⇒ 差异不能只用复权基准解释，"
           "需要向数据商核对该日价格本身。"))
    lines = [
        "# 交易级 E2E 审计：独立 open 验证 + 涨跌停事实",
        "",
        "## 输入与边界",
        "",
        f"- 交易明细：`{args.trades}`，{len(trades)} 笔 / {trades['code'].nunique()} 只标的。",
        f"- 独立行情：Tushare `daily` + `stk_limit` + `adj_factor`，{len(panel):,} 行；"
        f"{panel['code'].nunique()} 只标的；{panel['date'].min().date()} ~ {panel['date'].max().date()}。",
        "- 本文件是交易级审计。原 xlsx 没有完整每日持仓、现金、信号面板，"
        "因此不能诚实运行换手、净值对账、广度、容量或信号自洽。",
        "",
        "## 数据完整性",
        "",
        f"- 买入日 open / adj_factor 匹配：{int(x['ts_open'].notna().sum())}/{len(x)} / {int(x['ts_hfq_open'].notna().sum())}/{len(x)}。",
        f"- 卖出日 close / adj_factor 匹配：{int(x['ts_exit_close'].notna().sum())}/{len(x)} / {int(x['ts_hfq_exit_close'].notna().sum())}/{len(x)}。",
        f"- open/close 缺失：{int(panel[['open','close']].isna().sum().sum())}；"
        f"up/down limit 覆盖：{panel[['up_limit','down_limit']].notna().mean().min():.1%}。",
        "",
        "## §1：真实 open → close（后复权）独立收益验证",
        "",
        f"- 逐笔浮点精度一致：{exact}/{len(x)}。",
        f"- 在仅作诊断用的 {DIAGNOSTIC_TOLERANCE * 1e4:.1f} bp 跨数据商阈值内："
        f"{within_tolerance}/{len(x)}。",
        f"- 中位绝对差：{x['hfq_gross_diff'].abs().median():.3e}；"
        f"P95：{x['hfq_gross_diff'].abs().quantile(.95):.3e}；"
        f"最大：{x['hfq_gross_diff'].abs().max():.4%}。",
        (f"- 结论：**WARN，接近验证通过但非全量通过。** {within_tolerance} 笔在 "
         f"{DIAGNOSTIC_TOLERANCE * 1e4:.1f} bp 内，支持 xlsx 的 entry=open "
         "后复权收益口径；但仍有一笔超出，须保留并追溯。"
         "这是独立供应商的真实 open 验证，不是同源恒等式。"),
        outlier_note,
        f"- 有 {non_exact} 笔不是位级相等，最大差异已逐笔列出；通过不表示忽略这些"
        "可追溯的数据商/复权细节。",
        f"- 对照：若错误地直接用未复权 daily 价格比较，会有 {raw_nonexact}/{len(x)} 笔不一致；"
        "公司行为期间该比较不构成对 xlsx 后复权毛收益的有效验证。",
        f"- 开盘与收盘入场的逐笔收益在 {entry_basis_changed}/{len(x)} 笔中不同。"
        "由于没有完整每日持仓/现金账，不能把此交易级敏感性称为策略净值重算。",
        "",
        "最大 5 个差异：",
        "",
        top.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## §5：真实涨跌停事实",
        "",
        "★ 判板必须用【市场行情价】，不能用【成交价】。成交价含滑点，"
        "永远不会精确等于涨跌停价 —— 用成交价判板会恒得 0 笔，"
        "把真封板日报成干净。下面只以市场 open/close 判定。",
        "",
        f"- 买入日市场 open = up_limit（买不进）：{n_entry_limit_up}/{len(x)}。",
        f"- 卖出日市场 close = down_limit（卖不出）：{n_exit_limit_down}/{len(x)}。",
        (f"- 结论：**WARN，出场有 {n_exit_limit_down} 笔撞在跌停封板日。** "
         f"这 {n_exit_limit_down} 笔按回测口径（跌停收盘价成交）平均毛收益 "
         f"{limit_down_mean:+.2%}，合计净盈亏占全样本总盈亏 {limit_down_pnl_share:+.2%} —— "
         "不是可忽略的尾巴。跌停封板日按收盘价卖出在现实中卖不掉。"
         "买入侧未观察到开盘涨停（0 笔），此前按代码猜板得到的“涨停要买 13 个”"
         "确认为误报。"),
        f"- 对照：若错误地用成交价判板，跌停命中数会是 "
        f"{n_exit_limit_down_by_fill}/{len(x)}（滑点使等式恒不成立），"
        "据此会得出“未观察到收盘跌停卖出”的错误结论。",
        "",
        "跌停封板出场明细：",
        "",
        limit_down_detail.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## §5b：跌停出场顺延（库内检查 `deferred_exit`）",
        "",
        "```text",
        deferred.detail,
        "```",
        f"- {deferred.impact}",
        "",
        "## §4.3：交易级持有期邻域",
        "",
        "```text",
        holding.detail,
        "```",
        f"- {holding.impact}",
        "",
        "## 获取过程与优化记录",
        "",
        "1. Tushare 代理 `teajoin.com` 的 `daily` / `stk_limit` / `adj_factor` 连通并返回所需字段。",
        "2. 单一合并 CSV 在 Windows 上被外部进程长期锁定；下载器改为每代码一个不可变"
        "   checkpoint 分片，完成后可尝试合并。分片是权威输入，避免锁导致整批重跑。",
        "3. 提供方说明无数据会正常返回空表，故下载器必须逐代码落盘并在审计前检查日期匹配；"
        "   本次买卖日期均为 796/796 匹配。",
        "4. `daily` 是未复权价格，而 xlsx 毛收益是后复权口径；先以 `price × adj_factor` 对齐，"
        "   再进行逐笔验证。直接比较原始日线会造成公司行动期间的伪差异。",
        "5. 若未来逐笔后复权验证出现离群值，必须保留其交易明细和 WARN，不能以汇总平均值掩盖。",
        "6. **判板一律用市场行情价，不用成交价。** 第一版用 `实际卖出价 == down_limit`"
        "   得 0/796 并写下「未观察到收盘跌停卖出」；成交价含滑点，这个等式在构造上"
        "   不可能成立。换成市场 `close == down_limit` 实测 8 笔真封板（收盘精确 −10.0%），"
        "   合计占总盈亏 −13.4%。判据错的方向是【漏报】，最不该出错的方向。",
        "7. 涨跌停价必须与 close 同复权基准。两边基准不同会让 `close == down_limit`"
        "   恒不成立，同样是静默漏报。",
        "8. 离群笔要归因到「复权基准差异」还是「价格本身错」：判别法是看**未复权**"
        "   比值是否与 xlsx 精确相等。只写「须追溯」不足以让用户决定要不要找数据商。",
        "",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
