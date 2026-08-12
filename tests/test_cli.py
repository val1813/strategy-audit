"""CLI 端到端。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from strategy_audit import core
from strategy_audit.cli import main

from synth import equal_weight, lookahead_weight, make_prices, month_ends, to_long


@pytest.fixture()
def files(tmp_path, px, wm_clean):
    wp = tmp_path / "w.csv"
    pp = tmp_path / "p.csv"
    to_long(wm_clean, "weight").to_csv(wp, index=False)
    to_long(px, "close").to_csv(pp, index=False)
    return str(wp), str(pp), tmp_path


def test_cli_clean_exits_zero(files, capsys):
    wp, pp, _ = files
    assert main(["--weights", wp, "--prices", pp]) == 0
    out = capsys.readouterr().out
    assert "策略审计" in out and "换手与成本" in out


def test_cli_dirty_exits_one(tmp_path, px, wm_dirty, capsys):
    wp, pp = tmp_path / "w.csv", tmp_path / "p.csv"
    to_long(wm_dirty, "weight").to_csv(wp, index=False)
    to_long(px, "close").to_csv(pp, index=False)
    assert main(["--weights", str(wp), "--prices", str(pp)]) == 1
    assert "BLOCK" in capsys.readouterr().out


def test_cli_warn_does_not_fail_ci(tmp_path, px, capsys):
    """★ WARN 不该挂 CI —— 否则客户会去关告警。"""
    reb = month_ends(px)
    codes = list(px.columns[:10])
    wm = pd.DataFrame(0.1, index=pd.Index(reb, name="date"), columns=codes)
    wp, pp = tmp_path / "w.csv", tmp_path / "p.csv"
    to_long(wm, "weight").to_csv(wp, index=False)
    to_long(px[codes], "close").to_csv(pp, index=False)
    code = main(["--weights", str(wp), "--prices", str(pp)])
    out = capsys.readouterr().out
    assert "⚠" in out
    assert code == 0


def test_cli_missing_file_exits_cleanly(files):
    wp, pp, _ = files
    with pytest.raises(SystemExit, match="文件不存在"):
        main(["--weights", "nope.csv", "--prices", pp])


def test_cli_net_series_reconciles(files, px, wm_clean, capsys):
    wp, pp, tmp = files
    wmn = wm_clean.div(wm_clean.abs().sum(axis=1), axis=0).fillna(0.0)
    pr = core.period_returns(wmn, px)
    to = core.turnover(wmn, px)
    t = to["drift_adj"].reindex(pr.index).fillna(0.0)
    net = pr["ret"] - 2.0 * 18e-4 * t
    npth = tmp / "net.csv"
    pd.DataFrame({"date": net.index, "net": net.values}).to_csv(npth, index=False)
    assert main(["--weights", wp, "--prices", pp, "--net", str(npth)]) == 0
    out = capsys.readouterr().out
    assert "毛净对账" in out and "18" in out


def test_cli_net_requires_single_value_column(files, tmp_path):
    wp, pp, _ = files
    bad = tmp_path / "bad.csv"
    pd.DataFrame({"date": ["2021-01-04"], "a": [0.1], "b": [0.2]}).to_csv(
        bad, index=False)
    with pytest.raises(SystemExit, match="恰好一列"):
        main(["--weights", wp, "--prices", pp, "--net", str(bad)])


def test_cli_net_requires_date_column(files, tmp_path):
    wp, pp, _ = files
    bad = tmp_path / "bad.csv"
    pd.DataFrame({"t": ["2021-01-04"], "v": [0.1]}).to_csv(bad, index=False)
    with pytest.raises(SystemExit, match="date"):
        main(["--weights", wp, "--prices", pp, "--net", str(bad)])


def test_cli_reads_parquet(tmp_path, px, wm_clean):
    pytest.importorskip("pyarrow")
    wp, pp = tmp_path / "w.parquet", tmp_path / "p.parquet"
    to_long(wm_clean, "weight").to_parquet(wp)
    to_long(px, "close").to_parquet(pp)
    assert main(["--weights", str(wp), "--prices", str(pp)]) == 0


def test_cli_name_appears_in_report(files, capsys):
    wp, pp, _ = files
    main(["--weights", wp, "--prices", pp, "--name", "我的策略X"])
    assert "我的策略X" in capsys.readouterr().out
