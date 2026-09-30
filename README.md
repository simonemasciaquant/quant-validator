# quant-validator

**Stop overfitting. Validate your strategy before you trade it.**

A Python toolkit for rigorous validation of quantitative trading strategies.
Tests placebo, LOAO, walk-forward, and Deflated Sharpe Ratio on your trade series.

## The problem

95% of retail strategies fail in production. The reason: backtest overfitting.
A strategy that shows Sharpe 2.0 in backtest often shows 0.0 in production,
because the backtest was curve-fitted to the past.

quant-validator helps you detect overfitting BEFORE you risk capital.

## What it does

Given a trade series (asset, entry_ts, exit_ts, ret_net), it runs:

- Basic metrics: Sharpe (calendar + active), Sortino, Max DD, Calmar, PF, Hit rate
- Placebo test: shuffles direction, tests if signal has predictive power
- LOAO: leave-one-asset-out stability
- Walk-forward: IS/OOS split and rolling analysis
- Deflated Sharpe Ratio: corrects Sharpe for number of trials tested
- Random benchmark: compares to random entries in same window

## Install

    pip install -e .

## Quick start

    import pandas as pd
    from quant_validator import Validator

    trades = pd.read_csv('my_trades.csv')
    v = Validator(trades, capital=10000, max_conc=10)
    v.run_all(split_date='2024-01-01', n_trials=8)
    print(v.summary())

## Example output

    --- BASIC METRICS ---
      N obs (daily):         2298
      Sharpe (calendar):     0.738
      Sharpe (active):       1.933
      Max DD:                -8.57%
      WARNING: Sharpe active is 2.6x Sharpe calendar.

    --- DEFLATED SHARPE RATIO ---
      Sharpe (annual):       0.738
      DSR:                   0.0000
      FAIL (need > 0.95)

## License

MIT
