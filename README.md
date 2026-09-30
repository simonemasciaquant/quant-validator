# quant-validator

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/simonemasciaquant/quant-validator/blob/main/examples/quickstart.ipynb)

**Stop overfitting. Validate your strategy before you trade it.**

Most strategies that look great in backtest fail in production.
Not because they're wrong, but because they're curve-fitted to the past.

This toolkit runs the standard validation battery on your trade series
and tells you if your edge is real or noise.

**[Try it in Colab in 30 seconds](https://colab.research.google.com/github/simonemasciaquant/quant-validator/blob/main/examples/quickstart.ipynb)** - no installation required.

## Install

    pip install quant-validator

## 30-second demo

    import pandas as pd
    from quant_validator.core import Validator

    trades = pd.read_csv('my_trades.csv')
    v = Validator(trades, capital=10000, max_conc=10)
    v.run_all(split_date='2024-01-01', n_trials=8)
    print(v.summary())

You get a single, opinionated report:

    --- BASIC METRICS ---
      Sharpe (calendar):     0.738
      Sharpe (active):       1.933
      Max DD:                -8.57%
      PF:                    1.456
      Hit rate:              56.08%

      WARNING: Sharpe active is 2.6x Sharpe calendar.

    --- PLACEBO TEST ---
      Real PF:               1.932
      Placebo mean PF:       1.002
      Percentile:            100.0%
      >>> PASS (need > 95%)

    --- LOAO ---
      Min Sharpe:            0.613
      >>> PASS (need min > 0.5)

    --- WALK-FORWARD ---
      Stability ratio:       0.605
      >>> PASS (need ratio > 0.5)

    --- DEFLATED SHARPE RATIO ---
      DSR:                   0.0000
      >>> FAIL (need > 0.95)

Interpretation: the strategy has a real edge (placebo, LOAO, walk-forward pass),
but after correcting for the number of trials tested, the Sharpe is not
statistically significant (DSR fails). Not tradeable.

## What it tests

| Test | What it detects |
|---|---|
| Sharpe (calendar) | True Sharpe including days with no positions |
| Sharpe (active) | Conditional Sharpe (days with positions only) |
| Placebo | Whether the signal direction has predictive power |
| LOAO | Whether the edge is concentrated in one asset |
| Walk-forward | Whether the strategy survives out-of-sample |
| Deflated Sharpe Ratio | Whether the Sharpe is significant after multiple testing |
| Random benchmark | Whether timing beats random entries in same window |

## Why it matters

The toolkit implements methods from:

- Bailey & Lopez de Prado (2014) - The Deflated Sharpe Ratio
- Lopez de Prado (2018) - Advances in Financial Machine Learning
- Harvey & Liu (2015) - Backtesting

These are the same methods used by institutional quants. They're not
proprietary. They're just not packaged for individual researchers.

## Design principles

1. Opinionated verdict. Not just numbers. Pass or fail per test.
2. No hidden parameters. Every default has a justification.
3. One function call. `v.run_all()` runs everything.
4. Calendar and active Sharpe separated. Most tools conflate them.
5. DSR included by default. Because it's the one test that catches
   the most common mistake.

## Use cases

- Retail traders: validate your EA before risking capital
- Prop firm traders: verify robustness before challenges
- Quant researchers: standard validation battery for papers
- Small funds: pre-diligence check on strategies

## Install

    pip install quant-validator

Or from source:

    git clone https://github.com/simonemasciaquant/quant-validator.git
    cd quant-validator
    pip install -e .

## Requirements

- Python 3.8+
- numpy, pandas, scipy

## Roadmap

- [x] Core validation battery (Sharpe, placebo, LOAO, walk-forward, DSR)
- [x] Quickstart notebook
- [ ] HTML report with charts
- [ ] Portfolio-level validation
- [ ] Integration with backtest frameworks
- [ ] Case studies from real strategies

## License

MIT - see LICENSE.

---

Note: this toolkit does NOT confirm that a strategy will be profitable.
It only tests whether the backtest is statistically sound. A strategy can
pass all tests and still lose money. Use at your own risk.
