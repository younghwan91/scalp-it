<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/images/brand/wordmark-dark.svg">
  <img alt="scalp-it" src="docs/images/brand/wordmark-light.svg" height="52">
</picture>

# scalp-it

**A pre-registered validation framework and real-time tick/orderbook runner for Korean equity (KOSPI · KOSDAQ) intraday scalping strategies.**

Live order placement has been halted since 2026-09-16; only the dry-run detector and intraday collection cron jobs are active.

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

[Guide](docs/guide.en.md) · [Verdict Log](docs/TRADING_CRITERIA.md) · [Runbook](docs/PAIR_TRADING_RUNBOOK.md) · [한국어](README.md)

</div>

> [!NOTE]
> Despite the repo name, it no longer targets swing trading—it **replicates a public individual trader's intraday scalping system**.
> The October 2026 pre-registered research program was **officially concluded with all forward tests rejected** (failing to clear retail round-trip costs).
> Current verdicts and decisions are recorded in [`docs/TRADING_CRITERIA.md`](docs/TRADING_CRITERIA.md).

## Try it now

No Kiwoom account, DB or API key needed. This reads the committed pre-registration lock files (`research/runs/*/FINAL.lock`) and prints the 10 forward verdicts (read-only).

```bash
git clone https://github.com/younghwan91/scalp-it && cd scalp-it
uv sync --extra dev
uv run python scripts/forward_verdicts.py   # all forward verdicts
uv run pytest -q                            # unit tests
```

## Key Results

Negative results are published as they are. Every number is backed by a document in this repo.

- **All 10 pre-registered forward verdicts rejected**: single-shot mean -5.9 to -44.7 bp, every CI lower bound negative. [Decision log](docs/TRADING_CRITERIA.md) (Korean).
- **Cost wall**: ~41.5 bp retail round trip (tax 20 + fees 3 + spread 14-31). Even an oracle that picks the exit perfectly loses -35.9 bp on 10-second market-order scalps. [One-page summary](docs/research/manju/99-where-we-stand.md) (Korean).
- **Universe diagnosis**: over 10 years / 5.5M daily observations, the top-300 by prior-day turnover lose -28.0 bp intraday; gains exist only overnight. Same document.
- **Overnight selection (autoresearch) rejected**: +45.3 bp gross, -18.0 bp net per trade. [VERDICT](research/autoresearch/VERDICT.md).

## Key Features

- **Single-session collection + real-time detection**: Streams Kiwoom WebSocket (`0B` trades · `0D` orderbook) frames into TimescaleDB while simultaneously feeding `PairDetector` to evaluate leader–follower pair signals with near-zero latency.
- **Strict guardrails & dry-run default**: Execution only proceeds after passing `RiskGuard` (daily loss limit, consecutive stop-loss limit, kill switch) and the shared `krx-quant-core` order guard; without `--confirm-real`, zero orders are placed (dry-run).
- **Pre-registration & single-test discipline**: Rejected hypotheses are never rescued by post-hoc parameter tuning; all verdicts and decisions are logged chronologically in [`docs/TRADING_CRITERIA.md`](docs/TRADING_CRITERIA.md).

## Setup & Usage

Running live collection and database scripts requires the following environment:

- **Kiwoom Securities account & API keys**: Obtain App key/secret and access them via sibling repo [kiwoom-client](https://github.com/younghwan91/kiwoom-client).
- **Sibling repo `quant-airflow`**: Clone it in the same parent directory. `KR_QUANT_DB`, `KIWOOM_APP_KEY`, `KIWOOM_APP_SECRET`, and `DART_API_KEY` are loaded from its `.env` (never stored in this repo).
- **Held-stock blocklist**: The `--confirm-real` path rejects startup if `SCALP_HELD_STOCK_CODES` (comma-separated ticker codes held in the account) is empty (fail-closed).

```bash
export $(grep -v '^#' ../quant-airflow/.env | xargs)                   # KR_QUANT_DB + KIWOOM_APP_*
```

**Intraday real-time** — The commands below hit the live API and write to the live DB. Since cron already runs them, only run manually when avoiding duplicate App key sessions.

```bash
uv run --extra pg scalp-pair-detect --live --live-order --duration 09:00-15:20   # Pair detector (dry-run without --confirm-real)
uv run scalp-tick-collect --auto 45 --duration 08:50-15:40                        # Tick & orderbook collector daemon (0B/0D)
uv run scalp-tick-health --minutes 20                                             # Collection health check (read-only)
```

**Batch & verification**

```bash
uv run scalp-theme-snapshot --sleep 0.3                      # Today's Kiwoom theme membership snapshot
uv run scalp-intraday --codes 005930 --pages 5               # Minute-bar collection
uv run --extra pg python scripts/tick_sanity.py --date today # Tick sanity check
python3 scripts/pair_daily_summary.py                        # Daily pair dry-run performance summary
```

## Caveats

- **Live orders halted**: Live order placement has been stopped since the night of 2026-09-16. Only between 2026-08-27 and 2026-09-16 were 1-share live orders placed via `--confirm-real`; currently 0 orders are sent.
- **One WebSocket session per App key**: The pair detector also collects ticks/orderbooks; launching `scalp-tick-collect` simultaneously kicks the active session and drops data.
- **DB dependency**: Ticks and orderbooks cannot be backfilled retroactively. If TimescaleDB PRIMARY (`KR_QUANT_DB`) goes down during market hours, that day's data is permanently lost ([details](docs/guide.en.md#db-dependency-caution)).
- **Discipline & costs**: Pre-register, measure once, and record outcomes as-is. Inspect median, win rate, payoff ratio, quantiles, and year-by-year breakdowns together, and deduct 2026 round-trip costs (34 bp) first. Never commit data files.

## Documentation

| Doc | Description |
|---|---|
| [`docs/guide.en.md`](docs/guide.en.md) | Architecture, weekday cron operations, DB dependency, `krx-quant-core` symbol mapping |
| [`docs/TRADING_CRITERIA.md`](docs/TRADING_CRITERIA.md) | **Start here** — Chronological decisions and current verdict status |
| [`docs/PROMOTION_CHECKLIST.md`](docs/PROMOTION_CHECKLIST.md) | 11 gates required before resuming live orders |
| [`docs/PAIR_TRADING_RUNBOOK.md`](docs/PAIR_TRADING_RUNBOOK.md) | Pair live-trading runbook — dry-run → live order procedure, safeguards |
| [`docs/process/daily-ops.html`](docs/process/daily-ops.html) | Weekday cron process graph by time and role |
| `docs/research/manju/` | Original system research, pre-registration, and verdicts |
| `docs/archive/` | v1/v2 historical records + 5 failures to never forget |

## More from the same author

| Repo | Role |
|---|---|
| [kiwoom-client](https://github.com/younghwan91/kiwoom-client) | Quotes, flow, minute bars, themes, orders (182 REST + WebSocket) |
| [krx-quant-core](https://github.com/younghwan91/krx-quant-core) | Korean equity core — tick size, price limits, sessions, order guards, fill logic, validation stats ([mapping table](docs/guide.en.md#shared-core-dependency-krx-quant-core)) |
| [fin-checkup](https://github.com/younghwan91/fin-checkup) | Free traffic-light reading of listed-company financials; 17 metrics with formulas and thresholds, measurements only |
| [daytrade-it](https://github.com/younghwan91/daytrade-it) | Independent research record measuring whether news day-trading strategies have a real edge |
| [swing-it](https://github.com/younghwan91/swing-it) | Validation framework (walk-forward · negative controls) + migrated low-frequency factors |
| [quant-airflow](https://github.com/younghwan91/quant-airflow) | Daily data ingestion + TimescaleDB PRIMARY/replica setup ([DB caveats](docs/guide.en.md#db-dependency-caution)) |

## License

[MIT](LICENSE)
