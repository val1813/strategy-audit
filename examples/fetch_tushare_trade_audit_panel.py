"""Download an independent open/close/limit panel for a trade-level audit.

The API token comes from ``$TUSHARE_TOKEN``, or from the file named by
``--token-file`` / ``$TUSHARE_TOKEN_FILE``.  It is never written to the
repository or to any output file.  Output is kept outside the package's
tracked sources and never overwrites a user's own close panel: this audit
needs an independent, unmodified market-data source.
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import pandas as pd
import tushare as ts


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "data" / "trade_audit_open_limits.csv"
START, END = "20200101", "20260811"
SLEEP, RETRIES = 0.22, 3


def read_token(token_file: Path | None) -> str:
    """Token from $TUSHARE_TOKEN, else --token-file / $TUSHARE_TOKEN_FILE."""
    token = os.environ.get("TUSHARE_TOKEN", "").strip()
    if token:
        return token
    path = token_file or (Path(os.environ["TUSHARE_TOKEN_FILE"])
                          if os.environ.get("TUSHARE_TOKEN_FILE") else None)
    if path is None:
        raise SystemExit(
            "no Tushare token: set $TUSHARE_TOKEN, or pass --token-file / "
            "set $TUSHARE_TOKEN_FILE to a file containing it")
    token = Path(path).read_text(encoding="utf-8").strip()
    if not token:
        raise SystemExit(f"Tushare token file is empty: {path}")
    return token


def client(token: str):
    ts.set_token(token)
    pro = ts.pro_api()
    pro._DataApi_token = token
    pro._DataApi__http_url = "https://teajoin.com"
    return pro


def codes_from_xlsx(path: Path) -> list[str]:
    d = pd.read_excel(path, sheet_name="交易明细", usecols=["内部代码"])
    raw = d["内部代码"].dropna().astype(str).str.lower().str.strip()
    out = []
    for s in raw:
        if len(s) != 8 or s[:2] not in {"sz", "sh", "bj"}:
            raise ValueError(f"无法从内部代码转换 Tushare 代码: {s!r}")
        out.append(f"{s[2:]}.{s[:2].upper()}")
    return sorted(set(out))


def request(pro, api: str, code: str, fields: str) -> pd.DataFrame:
    last = None
    for attempt in range(RETRIES):
        try:
            return getattr(pro, api)(ts_code=code, start_date=START,
                                     end_date=END, fields=fields)
        except Exception as exc:  # network / provider limit: retry conservatively
            last = exc
            time.sleep(1.0 * (attempt + 1))
    raise RuntimeError(f"{api} {code} failed after {RETRIES} tries: {last}")


def part_dir(output: Path) -> Path:
    return output.with_name(output.stem + "_parts")


def existing_codes(output: Path) -> set[str]:
    """Use one immutable file per code: no shared append-file lock in Windows."""
    folder = part_dir(output)
    if not folder.exists():
        return set()
    return {p.stem.replace("_", ".") for p in folder.glob("*.csv")}


def factor_dir(output: Path) -> Path:
    return output.with_name(output.stem + "_adj_factor_parts")


def write_part(frame: pd.DataFrame, output: Path, code: str) -> None:
    folder = part_dir(output)
    folder.mkdir(parents=True, exist_ok=True)
    # Period is legal in Windows file names, but an underscore keeps the part
    # name friendly to globbing.  Write once: checkpoint files are immutable.
    frame.to_csv(folder / f"{code.replace('.', '_')}.csv", index=False, encoding="utf-8")


def write_factor(frame: pd.DataFrame, output: Path, code: str) -> None:
    folder = factor_dir(output)
    folder.mkdir(parents=True, exist_ok=True)
    frame.to_csv(folder / f"{code.replace('.', '_')}.csv", index=False, encoding="utf-8")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trades", type=Path, required=True,
                    help="workbook with a 交易明细 sheet holding 内部代码")
    ap.add_argument("--output", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--token-file", type=Path, default=None,
                    help="file holding the Tushare token (else $TUSHARE_TOKEN)")
    args = ap.parse_args(argv)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    done = existing_codes(args.output)
    codes = codes_from_xlsx(args.trades)
    pro = client(read_token(args.token_file))
    for i, code in enumerate(codes, 1):
        if code not in done:
            daily = request(pro, "daily", code, "ts_code,trade_date,open,close,amount")
            time.sleep(SLEEP)
            limits = request(pro, "stk_limit", code, "ts_code,trade_date,up_limit,down_limit")
            time.sleep(SLEEP)
            d = daily.merge(limits, on=["ts_code", "trade_date"], how="left")
            d = d.rename(columns={"ts_code": "code", "trade_date": "date"})
            d["date"] = pd.to_datetime(d["date"])
            d = d[["date", "code", "open", "close", "amount", "up_limit", "down_limit"]]
            write_part(d, args.output, code)
            print(f"daily [{i}/{len(codes)}] {code}: {len(d)} rows")
        fp = factor_dir(args.output) / f"{code.replace('.', '_')}.csv"
        if not fp.exists():
            factor = request(pro, "adj_factor", code, "ts_code,trade_date,adj_factor")
            time.sleep(SLEEP)
            factor = factor.rename(columns={"ts_code": "code", "trade_date": "date"})
            factor["date"] = pd.to_datetime(factor["date"])
            write_factor(factor[["date", "code", "adj_factor"]], args.output, code)
            print(f"factor [{i}/{len(codes)}] {code}: {len(factor)} rows")
    parts = sorted(part_dir(args.output).glob("*.csv"))
    if len(parts) == len(codes):
        merged = pd.concat((pd.read_csv(p) for p in parts), ignore_index=True)
        # A locked convenience CSV must not make the acquisition fail; parts
        # are authoritative and the audit loader accepts the directory.
        try:
            merged.to_csv(args.output, index=False, encoding="utf-8")
        except PermissionError:
            print(f"all {len(parts)} parts complete; combined CSV is locked, use {part_dir(args.output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
