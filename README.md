# strategy-audit

[![tests](https://github.com/val1813/strategy-audit/actions/workflows/tests.yml/badge.svg)](https://github.com/val1813/strategy-audit/actions/workflows/tests.yml)
[![python](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/downloads/)
[![license](https://img.shields.io/badge/license-MIT-green)](LICENSE)

**English** · [中文](README.zh-CN.md)

> **The backtest you ran is not the strategy you'll trade.**
> `strategy-audit` finds the gap — before real money does.

A local, offline audit for the most quietly expensive step in quantitative research: the journey from a table of *weights* to the *NAV* you report. It doesn't pick factors, it doesn't predict returns, and it never touches the network. It reads your files, recomputes your portfolio, and answers one question — loudly:

> **Is the NAV you're showing actually what these holdings, at these prices, on the days you could actually trade, would have produced?**

---

## The problem

You found a few factors, tuned the parameters, and the backtest annualizes nicely. The natural next step is Sharpe, drawdown, win rate, paper trading.

Slow down.

Most backtest failures aren't in the factors. They're in the **last mile** — the step that turns a list of holdings into a reported NAV:

- the weight table doesn't reconcile to the NAV you quoted;
- suspended stocks get quietly treated as if nothing happened;
- month-end selection quietly eats the month-end return;
- turnover and capacity look fine on paper, and fall apart on execution.

`strategy-audit` exists because all of these are measurable — and almost all of them are invisible in a Sharpe ratio.

## It has already caught

A monthly strategy (not written by this project) submitted weights, prices, and a self-reported NAV. The tool raised two BLOCKs.

**The first:** the self-reported NAV ran **4.6%** ahead of what the weights and prices actually recompute to. The cause — found only after reading the source — was daily rebalancing hidden inside a "monthly" equal-weight table via `ffill`. That return was never in the weight table.

**The second, and worse:** the same holdings, with suspended names and missing prices booked three reasonable ways, produced cumulative returns from **+99.1%** to **−3.9%** — a **103-percentage-point** spread. At that point, "is the factor any good" is the wrong question. The books aren't even closed.

There are **26 checks across 7 families**, but four deserve to be read first:

| Read first | The question it asks |
|---|---|
| **Self-reported NAV reconciliation** | Can the NAV you submitted be explained by the weights and prices you submitted? |
| **Missing-price accounting** | How far do suspensions, delistings, and missing prices move the result? |
| **Residual effective bets** | You think you hold 30 names — are you actually making one bet? |
| **Breakeven cost** | How many basis points of one-way cost until the edge is gone? |

The rest run too. They aren't a 26-item checkup for its own sake — they're there to tell you whether you can trust those four answers.

## Install

```bash
pip install strategy-audit
```

Requires Python 3.9+. **Zero data-source dependencies** — it reads your files and computes locally. Nothing is uploaded, nothing is downloaded.

## Quick start

Point it at a couple of files. Order doesn't matter — the tool detects each one and prints what it thinks it found, so check that first.

```bash
strategy-audit weights.csv prices.csv nav.csv
```

Only have a NAV curve? It still works:

```bash
strategy-audit nav.csv
```

No data yet? See what a report looks like:

```bash
strategy-audit --demo
```

Your *net* returns and *benchmark* can't be guessed from a filename — pass them explicitly:

```bash
strategy-audit weights.csv prices.csv \
  --net net_returns.csv \
  --benchmark csi300_returns.csv \
  --trials 20
```

`--trials 20` means *"this strategy was picked from roughly 20 factor/window/threshold combinations."* Don't treat it as decoration. The tool can't read your research history, so it discounts Sharpe by the number you give it.

Same thing from Python:

```python
from strategy_audit import audit

report = audit(weights, prices, nav, net=net_returns,
               benchmark=benchmark, n_trials=20)
print(report.text())
```

## Give it what you have

You don't need to reshape your data to a schema. `strategy-audit` auto-detects files and column names — including Chinese headers like 调仓日 / 证券代码 / 目标权重 / 收盘价 / 成交额. What you can audit scales with what you hand over:

| You have | Audited | Still missing |
|---|---|---|
| A NAV curve | **7 checks** — NAV quality and strategy-level significance | weights & prices |
| Weights + prices | **21 checks** — adds lookahead, turnover, accounting, risk, prescription | **5 checks (optional inputs)** — amount, net returns, self-reported NAV |
| + `amount` (turnover value) | **24 checks** — adds capacity, liquidity, size decay | net returns, self-reported NAV |
| Everything | **26 checks** — the full capability matrix | — |

"Couldn't check" is not the same as "passed." The report lists, check by check, exactly what's missing.

## The seven families

You don't need to memorize these — read the plain-language conclusions in the report. This is just so you know what each family is probing.

### NAV reconciliation *(runs first)*

Before any check trusts your numbers, it verifies the numbers are *yours*. If your reported NAV isn't the same curve as the one your weights and prices recompute to, then every check below is auditing a strategy you never actually traded.

### Lookahead & accounting

Did you eat the rebalance-day return? Are your weights carrying same-period information? Did your universe quietly keep only the survivors? How far do the missing-price policies move the answer?

### NAV quality

Is the curve itself even real? Smoothing and lagged pricing show up as return autocorrelation — understated volatility, overstated Sharpe. A NAV that sits still suggests valuations that aren't updating.

### Turnover & cost

Naive turnover vs. price-drifted turnover; whether autocorrelation-based turnover estimates are biased; the cost at which your edge goes to zero; and — if you provide gross and net — how much you actually paid.

### Strategy-level significance

Is the return concentrated in a couple of years or a few days? Does the conclusion flip when the Newey-West lag changes? How much Sharpe survives the multiple-testing discount?

### Risk identity

Is your name-count inflated? Thirty low-vol, small-cap, same-industry names look diversified but may be a handful of independent bets.

### Capacity & tradability

How much weight can't get filled at limit-up/down or on suspensions; how much money the portfolio can actually manage; whether the edge is concentrated in the least tradeable names. It reports execution risk — it doesn't sentence a strategy to death for being small.

### Prescription

The only family that says "here's how to change it" — and it's gated: it only suggests fixes that require *no forecasting*, like dropping weight-tweaks with no value. It refuses far more often than it prescribes, and "not prescribable" isn't a failure — it means the data can't support a clever-looking optimization.

## Reading the report

Three levels, no more:

| Mark | Meaning | What to do |
|---|---|---|
| **BLOCK** | Something key about this NAV doesn't add up | Go back to the code and data; fix it before discussing performance |
| **WARN** | The result still holds, but discount it by the reported magnitude | Check the size of the impact before paper trading or pitching |
| **OK** | This item was checked and nothing was found | It's about this one item — not a stamp on the whole strategy |

There's also a section called **"couldn't check,"** and it matters. No turnover-value column? Then capacity is reported as *not checked* — not *fine*.

A useful reading order:

1. Read the **BLOCKs** first — especially NAV reconciliation and the accounting-policy result.
2. Then breakeven cost and annualized one-way turnover. A strategy that needs 5 bp to be profitable is hard to trade in practice.
3. Then capacity. If the report says it manages tens of millions, don't explain a multi-billion sim with it.
4. Only then Sharpe, drawdown, prescription. If the books don't balance, precise statistics are precisely meaningless.

## What it will not do

- **Factor-level lookahead.** If you compute factors with future data and feed the result in as a clean equal-weight table, the weights and NAV can still reconcile — and the "weight lookahead" check reads as not-applicable for equal weights.
- **Count how many parameters you actually tried.** `n_trials` is on your honor; the tool can't read your backtest history.
- **Point at the exact line of code.** It can tell you *"the weight table can't explain your reported NAV"* and where to look, but without your strategy source it won't locate the bug.

## FAQ

**"I only have my platform's backtest chart — can I use this?"**
Yes. Export the dates and cumulative NAV and run the 7 checks. The result only says whether *that curve* has obvious problems.

**"Everything's OK — can I go live?"**
No. OK means the checked items found nothing. Unprovided data, capacity, real market impact, and future performance are all still outside the tool's scope.

**"There's a WARN — should I throw the strategy away?"**
Read what the WARN actually says. Capacity shortfall and month-end effect are handled differently; some need more data, some need you to discount annualization and Sharpe.

**"Why do you insist I hand over my self-reported NAV?"**
Because it's the check that catches the most dangerous bug. Weights and prices can recompute *a* curve, but the one your platform actually reported isn't necessarily the same one.

## Documentation

- **English** — this file
- **[中文](README.zh-CN.md)**

## Development

```bash
pip install -e ".[dev]"
pytest          # 358 tests
```

MIT License.
