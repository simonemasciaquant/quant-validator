# quant-validator

A rigorous validation toolkit for quantitative trading strategies.

`quant-validator` provides the statistical and methodological checks you
need before trusting a backtest: placebo tests, leave-one-asset-out
cross-validation, walk-forward analysis, deflated Sharpe ratio, event
clustering, and an account simulator with compounding size and a
chronological cap.

Built to accompany the eBook **Stop Overfitting** (v1.1).

---

## Install

    pip install git+https://github.com/simonemasciaquant/quant-validator.git

Or, for local development:

    git clone https://github.com/simonemasciaquant/quant-validator.git
    cd quant-validator
    pip install -e .

Dependencies: `pandas`, `numpy`. Python 3.9+.

---

## Quick start

    import pandas as pd
    from quant_validator import (
        build_events, simulate_account, Metrics,
    )

    ledger = pd.read_csv("ledger_trades.csv")
    ledger["entry_ts"] = pd.to_datetime(ledger["entry_ts"], utc=True)
    ledger["exit_ts"]  = pd.to_datetime(ledger["exit_ts"],  utc=True)

    # 1. Cluster trades into events (30-minute gap)
    clustered = build_events(ledger, gap_minutes=30)

    # 2. Simulate a compounding-size account, max 10 concurrent positions
    executed, equity, final_capital = simulate_account(
        ledger,
        capital_start=10_000,
        max_concurrent=10,
    )

    # 3. Standard metrics
    print("Trades executed:", len(executed))
    print("Final capital:  ", f"{final_capital:,.2f} EUR")
    print("Sharpe (calendar):", Metrics.sharpe_from_equity(equity))
    print("CAGR:             ", Metrics.cagr_from_equity(equity, 10_000))
    print("Max DD:           ", Metrics.max_dd_from_equity(equity))

---

## What's in the toolkit

| Component | What it does |
|---|---|
| `Metrics` | Sharpe, max drawdown, profit factor, and equity-based Sharpe / CAGR / DD |
| `Simulator` | Flat-sizing backtest engine |
| `PlaceboTest` | Randomization test against shuffled returns |
| `LOAO` | Leave-one-asset-out cross-validation |
| `WalkForward` | In-sample / out-of-sample split analysis |
| `DSR` | Deflated Sharpe ratio (Bailey & Lopez de Prado, corrected) |
| `Validator` | Orchestrator that runs the full validation battery |
| `build_events` | Temporal clustering of trades (default gap: 30 min) |
| `simulate_account` | Compounding-size account with chronological cap |

---

## Reference case: mean-reversion post-cascade

This is the case study of the eBook **Stop Overfitting** v1.1.

**Strategy.** Fade extreme moves (>3% in 30 minutes) during dead hours
(02:00-06:00 UTC), with a volume filter (Z-score > 2) and a regime
filter (BTC above its 200-day SMA on the previous day). Long-only.
Universe: 13 altcoins.

**Operational parameters.**

- Entry: at the close of the bar following the trigger (within 5 minutes)
- Target: 30% of `|ret_30m|`
- Stop: 30% of `|ret_30m|`
- Timeout: 4 hours
- Cap: maximum 10 concurrent positions
- Size: current capital / 10 (compounding)
- Costs: 4 bps round-trip

**Results (reference ledger v0.3.0).**

| Metric | Value |
|---|---|
| Executed trades | 1,635 |
| Events (gap = 30 min) | 426 |
| Sharpe (calendar) | 1.52 |
| CAGR | 7.06% |
| Max drawdown | -5.64% |
| Final equity | 15,357.51 EUR |
| Starting capital | 10,000 EUR |

> **Note.** The numbers published in *Stop Overfitting* v1.1 (1,633 trades,
> Sharpe 1.44, CAGR 6.35%, MaxDD -11.91%) refer to the previous data
> pipeline. The v0.3.0 toolkit regenerates the ledger from the corrected
> pipeline (no look-ahead regime, stop-first fill) and produces the
> numbers above.

---

## The five validation tests

| Test | What it catches |
|---|---|
| Calendar vs active Sharpe | Inflated Sharpe from non-contiguous trading |
| Placebo | Strategy with no real edge over random entries |
| LOAO | Over-reliance on a single asset |
| Walk-forward | In-sample overfitting |
| DSR | Multiple-testing inflation |

See `docs/methodology.md` for details.

---

## Reproducing the reference case

1. Regenerate the ledger with the corrected pipeline
   (no look-ahead regime, stop-first fill, 11 columns).
2. Run the quick-start snippet above.
3. Run the test suite:

    pytest tests/ -v

The reference regression test
(`tests/test_simulate.py::test_reference_ledger_numbers`) verifies
Sharpe, CAGR, MaxDD, and final equity against the ledger.

---

## License

MIT. See `LICENSE`.
