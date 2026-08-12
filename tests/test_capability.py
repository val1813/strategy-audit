"""能力矩阵：能审什么、审不了什么、补什么能多审几项。

★ 这一层的错会让用户做错决定
--------------------------
「补权重面板 ⇒ 多 1 项」和「多 7 项」会导致完全不同的行动。
第一版只试单个输入，把最有价值的建议（权重+价格一起补）藏了起来。
"""

from __future__ import annotations

import pytest

from strategy_audit import capability as cap


def test_nav_only_unlocks_significance_family():
    """★ 计数从 CHECKS 推导，不写死。

    新增一族检查时写死的数字会让一堆测试红掉，而那些测试跟新族毫无关系
    —— 实测加族四(风险身份)时 7 个测试因此失败。
    该断言的真意是「只有净值时解锁的恰好是族三」，那就直接这么断言。
    """
    ok, no = cap.available({cap.NAV})
    nav_only = [c for c in cap.CHECKS if set(c.needs) == {cap.NAV}]
    assert len(ok) == len(nav_only)
    assert all(c.section == "策略层显著性" for c in ok)
    assert len(no) == len(cap.CHECKS) - len(ok)


def test_weights_and_prices_unlock_almost_all():
    """权重+价格+净值 ⇒ 只剩需要净收益序列的那些审不了。"""
    ok, no = cap.available({cap.W, cap.P, cap.NAV})
    assert len(ok) == len(cap.CHECKS) - len(no)
    assert all(cap.NET in c.needs for c in no), [c.key for c in no]
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
    allkinds = {cap.W, cap.P, cap.NAV, cap.NET, cap.BENCH}
    ok, no = cap.available(allkinds)
    assert no == [], f"这些检查任何输入都解锁不了：{[c.key for c in no]}"


def test_check_keys_unique():
    keys = [c.key for c in cap.CHECKS]
    assert len(keys) == len(set(keys))


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
    assert cap.missing_value({cap.W, cap.P, cap.NAV, cap.NET, cap.BENCH}) == []


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
