"""End-to-end quick start for quant-validator v0.3.0."""

from pathlib import Path

import pandas as pd

from quant_validator import (
    build_events, simulate_account, Metrics,
)


LEDGER_PATH = Path(
    "/content/drive/MyDrive/cripto_altcoin/ebook/ledger_trades.csv"
)

ASSETS_FULL = [
    "AVAX", "DOT", "ATOM", "SOL", "LINK", "UNI", "AAVE",
    "NEAR", "FIL", "APT", "OP", "INJ", "LDO",
]

CAPITAL_START = 10_000
MAX_CONCURRENT = 10


def main():
    if not LEDGER_PATH.exists():
        raise SystemExit(f"Ledger not found at {LEDGER_PATH}")

    ledger = pd.read_csv(LEDGER_PATH)
    ledger["entry_ts"] = pd.to_datetime(ledger["entry_ts"], utc=True)
    ledger["exit_ts"]  = pd.to_datetime(ledger["exit_ts"],  utc=True)

    clustered = build_events(ledger, gap_minutes=30)
    n_events = clustered["event_id"].nunique()
    print(f"Events (gap=30min): {n_events}")

    executed, equity, final_capital = simulate_account(
        ledger,
        capital_start=CAPITAL_START,
        max_concurrent=MAX_CONCURRENT,
        assets_order=ASSETS_FULL,
    )

    sharpe = Metrics.sharpe_from_equity(equity)
    cagr   = Metrics.cagr_from_equity(equity, CAPITAL_START)
    dd     = Metrics.max_dd_from_equity(equity)

    print(f"Trades executed:    {len(executed):,}")
    print(f"Final capital:      {final_capital:,.2f} EUR")
    print(f"Sharpe (calendar):  {sharpe:.4f}")
    print(f"CAGR:               {cagr:.4%}")
    print(f"Max drawdown:       {dd:.4%}")


if __name__ == "__main__":
    main()
