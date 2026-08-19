"""CLI 端到端。位置参数 + 自动识别，不用记 flag。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from strategy_audit import core
from strategy_audit.cli import main

from synth import equal_weight, lookahead_weight, make_prices, month_ends, to_long


@pytest.fixture()
def files(tmp_path, px, wm_clean):
    wp, pp = tmp_path / "w.csv", tmp_path / "p.csv"
    to_long(wm_clean, "weight").to_csv(wp, index=False)
    to_long(px, "close").to_csv(pp, index=False)
    return str(wp), str(pp), tmp_path


def test_cli_positional_files_work(files, capsys):
    """★ 傻瓜式的核心：给文件就行，不用 --weights/--prices。"""
    wp, pp, _ = files
    assert main([wp, pp]) == 0
    out = capsys.readouterr().out
    assert "能审" in out and "换手与成本" in out


def test_cli_order_does_not_matter(files, capsys):
    """两个文件调换顺序 ⇒ 结论必须一样。"""
    wp, pp, _ = files
    main([wp, pp])
    a = capsys.readouterr().out
    main([pp, wp])
    b = capsys.readouterr().out
    # 除了识别明细里的「输入1/输入2」标签，其余结论应一致
    key = lambda s: [ln for ln in s.split("\n")
                     if ln.strip().startswith(("✅", "⚠", "❌"))]
    assert key(a) == key(b)


def test_cli_single_nav_file_still_audits(tmp_path, capsys):
    """★ 只给一条净值曲线也要能审（族三），不能报错退出。"""
    idx = pd.date_range("2016-01-31", periods=120, freq="ME")
    r = pd.Series(np.random.default_rng(11).normal(0.006, 0.045, 120), index=idx)
    f = tmp_path / "nav.csv"
    pd.DataFrame({"日期": idx, "累计净值": (1 + r).cumprod().values}).to_csv(
        f, index=False)
    code = main([str(f)])
    out = capsys.readouterr().out
    from strategy_audit import capability as cap
    assert "策略层显著性" in out
    assert f"审不了 {len(cap.available({cap.NAV})[1])} 项" in out
    assert code in (0, 1)


def test_cli_demo_needs_no_data(capsys):
    """★ 没有数据也能先看报告长什么样。"""
    code = main(["--demo"])
    out = capsys.readouterr().out
    from strategy_audit import capability as cap
    n_ok = len(cap.available({cap.W, cap.P, cap.NAV})[0])
    assert "内置合成数据" in out
    assert f"能审 {n_ok}/{len(cap.CHECKS)} 项" in out
    # 演示数据故意含缺陷（6 只退市股）⇒ 有 BLOCK ⇒ 退出码必须是 1。
    # 同一个工具在不同入口给不同的退出码语义是最难查的坑。
    assert "1 项 BLOCK" in out
    assert code == 1


def test_cli_no_args_points_to_demo(capsys):
    """没给参数时要引导，而不是抛栈。"""
    assert main([]) == 2
    assert "--demo" in capsys.readouterr().out


def test_cli_dirty_exits_one(tmp_path, px, wm_dirty, capsys):
    wp, pp = tmp_path / "w.csv", tmp_path / "p.csv"
    to_long(wm_dirty, "weight").to_csv(wp, index=False)
    to_long(px, "close").to_csv(pp, index=False)
    assert main([str(wp), str(pp)]) == 1
    assert "BLOCK" in capsys.readouterr().out


def test_cli_warn_does_not_fail_ci(tmp_path, px, capsys):
    """★ WARN 不该挂 CI —— 否则用户会去关告警。"""
    reb = month_ends(px)
    codes = list(px.columns[:10])
    wm = pd.DataFrame(0.1, index=pd.Index(reb, name="date"), columns=codes)
    wp, pp = tmp_path / "w.csv", tmp_path / "p.csv"
    to_long(wm, "weight").to_csv(wp, index=False)
    to_long(px[codes], "close").to_csv(pp, index=False)
    code = main([str(wp), str(pp)])
    out = capsys.readouterr().out
    assert "⚠" in out
    assert code == 0


def test_cli_missing_file_exits_cleanly(files):
    wp, pp, _ = files
    with pytest.raises(SystemExit, match="文件不存在"):
        main(["nope.csv", pp])


def test_cli_chinese_columns(tmp_path, px, wm_clean, capsys):
    """中文列名必须能认。"""
    wp, pp = tmp_path / "w.csv", tmp_path / "p.csv"
    to_long(wm_clean, "weight").rename(
        columns={"date": "调仓日期", "code": "证券代码",
                 "weight": "目标权重"}).to_csv(wp, index=False)
    to_long(px, "close").rename(
        columns={"date": "交易日", "code": "股票代码",
                 "close": "收盘价"}).to_csv(pp, index=False)
    main([str(wp), str(pp)])
    out = capsys.readouterr().out
    assert "权重面板" in out and "认不出来" not in out


def test_cli_reads_parquet(tmp_path, px, wm_clean):
    pytest.importorskip("pyarrow")
    wp, pp = tmp_path / "w.parquet", tmp_path / "p.parquet"
    to_long(wm_clean, "weight").to_parquet(wp)
    to_long(px, "close").to_parquet(pp)
    assert main([str(wp), str(pp)]) == 0


def test_cli_trials_flag_changes_discount(files, capsys):
    """--trials 必须真的进入多重检验折扣。"""
    wp, pp, _ = files
    main([wp, pp, "--trials", "50"])
    out = capsys.readouterr().out
    assert "50 个配置" in out


def test_cli_exposes_explicit_net_and_benchmark_roles(capsys):
    """文件用户也必须能传无法自动猜角色的净收益与基准。"""
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    out = capsys.readouterr().out
    assert exc.value.code == 0
    assert "--net" in out and "--benchmark" in out


def test_cli_name_appears_in_report(files, capsys):
    wp, pp, _ = files
    main([wp, pp, "--name", "我的策略X"])
    assert "我的策略X" in capsys.readouterr().out


def test_cli_detection_shown_by_default(files, capsys):
    """★ 识别结果默认必须打印 —— 认错了用户得能看出来。"""
    wp, pp, _ = files
    main([wp, pp])
    out = capsys.readouterr().out
    assert "输入识别结果" in out and "请核对" in out
