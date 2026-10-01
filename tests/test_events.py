"""Tests for build_events()."""
import pandas as pd
import pytest
from quant_validator import build_events


def _ts(s):
    return pd.Timestamp(s, tz="UTC")


def test_empty_dataframe():
    df = pd.DataFrame({"entry_ts": pd.Series([], dtype="datetime64[ns, UTC]")})
    out = build_events(df)
    assert len(out) == 0
    assert "event_id" in out.columns


def test_single_trade():
    df = pd.DataFrame({"entry_ts": [_ts("2024-01-01 02:00")]})
    out = build_events(df)
    assert out["event_id"].tolist() == [0]


def test_gap_below_threshold_same_event():
    df = pd.DataFrame({"entry_ts": [
        _ts("2024-01-01 02:00"), _ts("2024-01-01 02:15"), _ts("2024-01-01 02:30"),
    ]})
    out = build_events(df, gap_minutes=30)
    assert out["event_id"].tolist() == [0, 0, 0]


def test_gap_above_threshold_new_event():
    df = pd.DataFrame({"entry_ts": [
        _ts("2024-01-01 02:00"), _ts("2024-01-01 03:00"),
    ]})
    out = build_events(df, gap_minutes=30)
    assert out["event_id"].tolist() == [0, 1]


def test_exact_threshold_stays_same_event():
    df = pd.DataFrame({"entry_ts": [
        _ts("2024-01-01 02:00"), _ts("2024-01-01 02:30"),
    ]})
    out = build_events(df, gap_minutes=30)
    assert out["event_id"].tolist() == [0, 0]


def test_input_is_sorted_before_clustering():
    df = pd.DataFrame({"entry_ts": [
        _ts("2024-01-01 03:00"), _ts("2024-01-01 02:00"), _ts("2024-01-01 02:15"),
    ]})
    out = build_events(df, gap_minutes=30)
    assert out["entry_ts"].is_monotonic_increasing
    assert out["event_id"].tolist() == [0, 0, 1]


def test_realistic_cluster_from_ledger():
    df = pd.DataFrame({
        "asset": ["AVAX", "DOT", "ATOM", "SOL", "LINK", "UNI", "AAVE", "NEAR"],
        "entry_ts": [
            _ts("2024-04-13 02:15"), _ts("2024-04-13 02:15"),
            _ts("2024-04-13 02:15"), _ts("2024-04-13 02:20"),
            _ts("2024-04-13 02:20"), _ts("2024-04-13 03:45"),
            _ts("2024-04-13 03:50"), _ts("2024-04-14 04:30"),
        ],
    })
    out = build_events(df, gap_minutes=30)
    assert out["event_id"].tolist() == [0, 0, 0, 0, 0, 1, 1, 2]
    assert out["event_id"].dtype == "int64"


def test_input_not_mutated():
    df = pd.DataFrame({"entry_ts": [_ts("2024-01-01 02:00")]})
    _ = build_events(df)
    assert "event_id" not in df.columns
