# Changelog

All notable changes to `quant-validator` are documented here.
The format follows Keep a Changelog, and this project adheres to
Semantic Versioning.

## [0.3.0] - 2026-10-01

### Added

- **`build_events(trades_df, gap_minutes=30)`** - temporal clustering of
  trades into events. Consecutive trades with an entry-ts gap of 30
  minutes or less belong to the same event.
- **`simulate_account(trades_df, capital_start=10000, max_concurrent=10, assets_order=None)`**
  - compounding-size account simulator with a chronological cap.
  Size at each entry is `current_capital / max_concurrent`. Skips entries
  when the cap is saturated or when the same asset is already held.
- **`Metrics.sharpe_from_equity(equity_df, periods_per_year=365)`** -
  calendar Sharpe computed from an equity DataFrame (daily resampled,
  forward-filled).
- **`Metrics.cagr_from_equity(equity_df, capital_start)`** - compound
  annual growth rate from an equity DataFrame.
- **`Metrics.max_dd_from_equity(equity_df)`** - max drawdown from an
  equity DataFrame, computed on the daily resampled curve.
- **`LOAO`** - leave-one-asset-out cross-validation.
- **`WalkForward`** - in-sample / out-of-sample split analysis with
  stability ratio.
- Test suite: `tests/test_events.py` (8 tests) and
  `tests/test_simulate.py` (8 tests, including 2 regression tests
  against the reference ledger).

### Fixed

- **`simulate_account` - residual positions closed in the wrong order.**
  Positions still open at the end of the backtest were closed in
  insertion order instead of `exit_ts` order, producing a non-monotonic
  equity curve. Residual positions are now sorted by `exit_ts` before
  being closed.

### Changed

- `__init__.py` now exports `build_events`, `simulate_account`, `LOAO`,
  and `WalkForward`.
- README rewritten with a full quick-start example and the reference
  case numbers for v0.3.0.

### Removed

- **`RandomBenchmark`** - was mentioned in earlier planning documents
  but never implemented. Removed from the README to avoid confusion;
  `PlaceboTest` covers the same use case with a more rigorous design.

### Notes

- The reference regression test uses the ledger regenerated with the
  corrected pipeline (no look-ahead regime, stop-first fill, 11 columns).
  With the previous ledger (pre-fix), the numbers differ - see README.
- `LOAO` and `WalkForward` were present in the v0.2.0 source but were
  missing from the export list. Restored in this release.

## [0.2.0] - previous

- `Metrics`, `Simulator` (flat sizing), `PlaceboTest`, `DSR` (corrected),
  `Validator`.
- Published on GitHub, MIT license.
