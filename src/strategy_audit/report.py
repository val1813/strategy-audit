"""报告层：BLOCK / WARN / OK 三级 + 分组。

★ 沿用 factor-audit 的 PanelReport 形态（同样的三级、同样的
detail/impact 两行结构），这样两个工具的报告读起来是一套。
差别只有一处：策略审计的检查分族（换手成本 / 前视 / 成分变动），
所以 Finding 多一个 section 字段，text() 按族分段打印。

★ 为什么坚持 impact 字段必填才有价值
------------------------------------
"换手率 4.6x" 本身不是发现，"你按 1.98x 算成本，实际 4.6x，
差的这部分按 20bp 双边算吃掉年化 10.5pp" 才是发现。
没有 impact 的告警会被当成噪声划过去 —— factor-audit 实测如此。
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field

BLOCK = "BLOCK"      # 净值不可信，必须先修
WARN = "WARN"        # 可用但要打折
OK = "OK"
SKIP = "SKIP"          # 输入齐全但来源同一/数据质量不足，结论不可审

_ICON = {BLOCK: "❌", WARN: "⚠ ", OK: "✅", SKIP: "⏭ "}
_ORDER = {BLOCK: 0, WARN: 1, SKIP: 2, OK: 3}


@dataclass
class Finding:
    level: str
    name: str
    detail: str
    impact: str = ""
    section: str = ""
    # 对应 capability.CHECKS 的 key；输入识别/契约等非审计项保留空串。
    key: str = ""


@dataclass
class AuditReport:
    title: str = "策略审计"
    findings: list[Finding] = field(default_factory=list)
    stats: dict = field(default_factory=dict)
    # 检查未能执行的原因（缺列等），与"检查执行了但通过"区分开
    skipped: list[str] = field(default_factory=list)
    # 能力矩阵文本（capability.matrix_text 的结果），打在报告最前面
    capability: str = ""
    _active_key: str = field(default="", init=False, repr=False)

    def add(self, level, name, detail, impact="", section="", key=""):
        self.findings.append(Finding(level, name, detail, impact, section,
                                     key or self._active_key))

    @contextmanager
    def check(self, key: str):
        """给一次正式检查产生的 finding 标上 capability key。

        key 由顶层调度设定，检查函数本身无需散落重复的字符串；这让
        「矩阵声明不可用的检查绝不产生 finding」可以做全量断言。
        """
        previous = self._active_key
        self._active_key = key
        try:
            yield
        finally:
            self._active_key = previous

    def skip(self, name: str, why: str) -> None:
        """★ 跳过必须显式记录。

        静默跳过 = 客户以为查过了。factor-audit 的实测教训是
        「缺列降级为 WARN 而不是报错」，但降级也必须出现在报告里。
        """
        self.skipped.append(f"{name}：{why}")

    @property
    def blockers(self) -> list[Finding]:
        return [f for f in self.findings if f.level == BLOCK]

    @property
    def warnings(self) -> list[Finding]:
        return [f for f in self.findings if f.level == WARN]

    @property
    def trustworthy(self) -> bool:
        return not self.blockers

    def sections(self) -> list[str]:
        out: list[str] = []
        for f in self.findings:
            if f.section not in out:
                out.append(f.section)
        return out

    def text(self) -> str:
        w = 68
        lines = ["=" * w, f"  {self.title}", "=" * w, ""]

        # ★ 能力矩阵打在最前面：用户第一眼就该看到边界在哪，
        # 而不是读完报告才发现有八项根本没查。
        if self.capability:
            lines.append(self.capability)
            lines.append("")
            lines.append("-" * w)
            lines.append("")

        s = self.stats
        if s:
            head = []
            if "n_dates" in s:
                head.append(f"{s['n_dates']} 个调仓日")
            if "span" in s:
                head.append(str(s["span"]))
            if "n_names" in s:
                head.append(f"{s['n_names']} 只标的")
            if head:
                lines += [f"  {' / '.join(head)}", ""]

        for sec in self.sections():
            if sec:
                lines += [f"  【{sec}】", ""]
            group = [f for f in self.findings if f.section == sec]
            group.sort(key=lambda f: _ORDER[f.level])
            for f in group:
                lines.append(f"  {_ICON[f.level]} {f.name}")
                for ln in f.detail.split("\n"):
                    lines.append(f"       {ln}")
                if f.impact:
                    for i, ln in enumerate(f.impact.split("\n")):
                        lines.append(f"       {'⇒ ' if i == 0 else '   '}{ln}")
                lines.append("")

        if self.skipped:
            lines.append("  【未能检查】")
            lines.append("")
            for sk in self.skipped:
                lines.append(f"  ·  {sk}")
            lines.append("")
            lines.append("  ★ 以上各项【没有查过】，不是查过通过了。")
            lines.append("")

        n_b, n_w = len(self.blockers), len(self.warnings)
        if n_b:
            lines.append(f"  ★ {n_b} 项 BLOCK、{n_w} 项 WARN —— "
                         "这份净值的结论不可信，先修上面的 BLOCK")
        elif n_w:
            lines.append(f"  ⚠  无 BLOCK，{n_w} 项 WARN —— "
                         "结论可用，但按上面的量级打折")
        else:
            lines.append("  ✅ 无 BLOCK / WARN")
        return "\n".join(lines)
