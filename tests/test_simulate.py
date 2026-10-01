"""Tests for simulate_account()."""
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from quant_validator import simulate_account, Metrics, build_events


LEDGER_PATH = Path("/content/drive/MyDrive/cripto_altcoin/ebook/ledger_trades.csv")

ASSETS_FULL = [
    "AVAX", "DOT", "ATOM", "SOL", "LINK", "UNI", "AAVE",
    "NEAR", "FIL", "APT", "OP", "INJ", "LDO",
]


def _ts(s):
    return pd.Timestamp(s, tz="UTC")


def test_empty_input():
    df = pd.DataFrame(columns=["entry_ts", "exit_ts", "asset", "ret_net"])
    ex, eq, cap = simulate_account(df, capital_start=10000)
    assert len(ex) == 0
    assert len(eq) == 0
    assert cap == 10000.0


def test_single_trade_compounding_size():
    df = pd.DataFrame({
        "entry_ts": [_ts("2024-01-01 02:00")],
        "exit_ts":  [_ts("2024-01-01 03:00")],
        "asset":    ["AAA"],
        "ret_net":  [0.10],
    })
    ex, eq, cap = simulate_account(df, capital_start=10000, max_concurrent=10)
    assert len(ex) == 1
    assert ex.iloc[0]["size"] == pytest.approx(1000.0)
    assert ex.iloc[0]["pnl"]  == pytest.approx(100.0)
    assert cap == pytest.approx(10100.0)


def test_chronological_cap_skips_excess():
    df = pd.DataFrame({
        "entry_ts": [_ts("2024-01-01 02:00")] * 3,
        "exit_ts":  [_ts("2024-01-01 03:00")] * 3,
        "asset":    ["AAA", "BBB", "CCC"],
        "ret_net":  [0.01, 0.01, 0.01],
    })
    ex, _, _ = simulate_account(df, capital_start=10000, max_concurrent=2)
    assert len(ex) == 2


def test_no_same_asset_double_exposure():
    df = pd.DataFrame({
        "entry_ts": [_ts("2024-01-01 02:00"), _ts("2024-01-01 02:30")],
        "exit_ts":  [_ts("2024-01-01 04:00"), _ts("2024-01-01 04:30")],
        "asset":    ["AAA", "AAA"],
        "ret_net":  [0.01, 0.01],
    })
    ex, _, _ = simulate_account(df, capital_start=10000, max_concurrent=10)
    assert len(ex) == 1


def test_compounding_after_winning_trade():
    df = pd.DataFrame({
        "entry_ts": [_ts("2024-01-01 02:00"), _ts("2024-01-01 05:00")],
        "exit_ts":  [_ts("2024-01-01 03:00"), _ts("2024-01-01 06:00")],
        "asset":    ["AAA", "BBB"],
        "ret_net":  [0.10, 0.10],
    })
    ex, _, cap = simulate_account(df, capital_start=10000, max_concurrent=10)
    assert ex.iloc[1]["size"] == pytest.approx(1010.0)
    assert cap == pytest.approx(10201.0)


def test_residual_positions_closed_in_exit_ts_order():
    df = pd.DataFrame({
        "entry_ts": [_ts("2024-01-01 02:00"), _ts("2024-01-01 02:05")],
        "exit_ts":  [_ts("2024-01-01 05:00"), _ts("2024-01-01 04:00")],
        "asset":    ["AAA", "BBB"],
        "ret_net":  [0.10, 0.05],
    })
    ex, eq, _ = simulate_account(df, capital_start=10000, max_concurrent=10)
    assert eq["ts"].is_monotonic_increasing
    assert eq["equity"].iloc[-1] == pytest.approx(10150.0)


@pytest.mark.skipif(
    not LEDGER_PATH.exists(),
    reason="reference ledger not available (Colab-only path)",
)
def test_reference_ledger_numbers():
    ledger = pd.read_csv(LEDGER_PATH)
    ledger["entry_ts"] = pd.to_datetime(ledger["entry_ts"], utc=True)
    ledger["exit_ts"]  = pd.to_datetime(ledger["exit_ts"],  utc=True)

    ex, eq, cap = simulate_account(
        ledger, capital_start=10000, max_concurrent=10,
        assets_order=ASSETS_FULL,
    )
    print(f"\n[REGRESSION] n_exec={len(ex)} cap={cap:.2f}")

    sharpe = Metrics.sharpe_from_equity(eq)
    cagr   = Metrics.cagr_from_equity(eq, capital_start=10000)
    dd     = Metrics.max_dd_from_equity(eq)

    print(f"[REGRESSION] sharpe={sharpe:.4f} cagr={cagr:.4f} dd={dd:.4f}")

    assert len(ex) == 1635, f"Expected 1635 executed trades, got {len(ex)}"
    assert cap == pytest.approx(15357.51, rel=1e-3)
    assert sharpe == pytest.approx(1.5189, abs=1e-2)
    assert cagr   == pytest.approx(0.0706, abs=1e-3)
    assert dd     == pytest.approx(-0.0564, abs=1e-3)


@pytest.mark.skipif(
    not LEDGER_PATH.exists(),
    reason="reference ledger not available (Colab-only path)",
)
def test_reference_ledger_clusters():
    ledger = pd.read_csv(LEDGER_PATH)
    ledger["entry_ts"] = pd.to_datetime(ledger["entry_ts"], utc=True)
    ledger["exit_ts"]  = pd.to_datetime(ledger["exit_ts"],  utc=True)

    clustered = build_events(ledger, gap_minutes=30)
    n_clusters = clustered["event_id"].nunique()
    print(f"\n[REGRESSION] n_clusters={n_clusters}")
    assert n_clusters == 426, f"Expected 426 clusters, got {n_clusters}"
