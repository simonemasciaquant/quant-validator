# Methodology

## Why validation matters

Most quantitative strategies fail in live trading despite good backtest results.

## Tests implemented

### 1. Basic metrics

- Sharpe (calendar): uses all days including zero-return days.
- Sharpe (active): uses only days with positions.

Rule of thumb: always report Sharpe (calendar) for commercial evaluation.
If ratio active/calendar > 2, the strategy spends most time flat.

### 2. Placebo test

Shuffles the direction of each trade and recomputes PF.

### 3. Leave-one-asset-out (LOAO)

Iteratively excludes each asset and recomputes Sharpe.

### 4. Walk-forward

Splits into in-sample and out-of-sample. Stability ratio = OOS/IS Sharpe.

### 5. Deflated Sharpe Ratio (DSR)

Bailey & Lopez de Prado (2014).

## References

- Bailey & Lopez de Prado (2014). The Deflated Sharpe Ratio.
- Lopez de Prado (2018). Advances in Financial Machine Learning.
