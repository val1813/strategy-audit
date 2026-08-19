"""能力矩阵：能审什么、审不了什么、补什么能多审几项。

★ 这一层的错会让用户做错决定
--------------------------
「补权重面板 ⇒ 多 1 项」和「多 7 项」会导致完全不同的行动。
第一版只试单个输入，把最有价值的建议（权重+价格一起补）藏了起来。
"""

from __future__ import annotations

import pytest

from strategy_audit import capability as cap


def test_nav_only_unlocks_curve_only_families():
    """★ 计数与族名都从 CHECKS 推导，不写死。

    新增一族检查时写死的数字会让一堆测试红掉，而那些测试跟新族毫无关系
    —— 实测加族四（风险身份）时 7 个测试因此失败，加族七（净值质量）
    时这个测试又因为写死了「恰好是族三」而失败。

    真意是「只有净值时解锁的，恰好是那些只需要净值的检查」——
    那就直接这么断言，不点名具体是哪几族。
    """
    ok, no = cap.available({cap.NAV})
    nav_only = [c for c in cap.CHECKS if set(c.needs) == {cap.NAV}]
    assert len(ok) == len(nav_only)
    assert {c.key for c in ok} == {c.key for c in nav_only}
    assert len(no) == len(cap.CHECKS) - len(ok)
    # 只要一条曲线就能审的项必须【不止】显著性一族 ——
    # 族三默认曲线本身是真的，族七审的正是那个前提。
    assert len({c.section for c in ok}) >= 2, [c.section for c in ok]


def test_weights_and_prices_unlock_almost_all():
    """权重+价格+净值 ⇒ 剩下审不了的每一项都【只】缺可选输入。

    ★ 断言必须从 CHECKS 推导，不能写死「只缺 net」。
    第一版写死了 cap.NET，于是族五（容量）引入成交额列 AMT 之后
    这个测试红了 —— 而它跟族五毫无关系。真意是「基础输入齐了以后，
    剩下的都只缺可选输入」，那就直接这么断言。
    """
    base = {cap.W, cap.P, cap.NAV}
    # OWN_NAV 也是可选输入：NAV 可以是工具自己重算的，
    # 而「客户亲手交上来的那条净值」只有他给了才有。
    optional = {cap.NET, cap.BENCH, cap.AMT, cap.OWN_NAV}
    ok, no = cap.available(base)
    assert len(ok) == len(cap.CHECKS) - len(no)
    for c in no:
        lack = set(c.needs) - base
        assert lack and lack <= optional, (c.key, lack)
    assert "reconcile" in [c.key for c in no]


def test_nothing_unlocks_nothing():
    ok, no = cap.available(set())
    assert ok == []
    assert len(no) == len(cap.CHECKS)


def test_every_check_is_reachable():
    """★ 每项检查都必须能被某个输入组合解锁。

    不可达的检查是登记表写错了 —— 它永远不会跑，而能力矩阵会一直
    把它列在「审不了」里，用户永远看不懂要补什么。
    """
    # ★ 这里必须用【所有】已声明的输入种类，不能手写清单 ——
    # 手写清单会在新增输入种类时把「不可达」误报成正常。
    allkinds = {k for c in cap.CHECKS for k in c.needs}
    ok, no = cap.available(allkinds)
    assert no == [], f"这些检查任何输入都解锁不了：{[c.key for c in no]}"
    # 每个种类都必须真的被 _LABEL 收录，否则报告会漏出内部键名
    for k in allkinds:
        assert cap.label(k) != k, f"{k} 没有人类可读名字"


def test_check_keys_unique():
    keys = [c.key for c in cap.CHECKS]
    assert len(keys) == len(set(keys))


def test_can_run_uses_the_same_needs_as_the_capability_matrix():
    """调度判据与矩阵必须同源，避免「运行了却显示审不了」。"""
    for check in cap.CHECKS:
        have = set(check.needs)
        assert cap.can_run(check.key, have)
        for required in set(check.needs):
            assert not cap.can_run(check.key, have - {required})


def test_missing_value_reports_combo_not_just_single():
    """★ 只有净值时，最有价值的建议是「权重+价格一起补」（多 7 项）。

    第一版只试单个输入，用户看到「多 1 项」就不会去补了。
    """
    gains = cap.missing_value({cap.NAV})
    assert gains, "应当给出补什么的建议"
    best_need, best_delta, _ = gains[0]
    assert best_delta >= 7, f"最佳建议只解锁 {best_delta} 项，太少"
    assert cap.W in best_need and cap.P in best_need


def test_missing_value_sorted_by_payoff():
    gains = cap.missing_value({cap.NAV})
    deltas = [d for _, d, _ in gains]
    assert deltas == sorted(deltas, reverse=True)


def test_missing_value_empty_when_all_present():
    """什么都给齐了就不该再建议补东西。★ 种类清单从 CHECKS 推导。"""
    allkinds = {k for c in cap.CHECKS for k in c.needs}
    assert cap.missing_value(allkinds) == []


def test_matrix_text_distinguishes_cannot_from_passed():
    """★ 报告必须写明「审不了 ≠ 查过通过」。"""
    txt = cap.matrix_text({cap.NAV})
    ok, no = cap.available({cap.NAV})
    assert f"能审 {len(ok)}/{len(cap.CHECKS)} 项" in txt
    assert f"审不了 {len(no)} 项" in txt
    assert "【不是】查过通过" in txt


def test_matrix_text_never_leaks_internal_keys():
    """报告里不许出现 nav/weights 这种内部键名。"""
    txt = cap.matrix_text({cap.NAV})
    for raw in (" nav", " weights", " prices", " net\n"):
        assert raw not in txt, f"泄漏了内部键名 {raw!r}"


def test_label_is_human_readable():
    for k in (cap.W, cap.P, cap.NAV, cap.NET, cap.BENCH):
        lb = cap.label(k)
        assert lb != k and len(lb) > 3
