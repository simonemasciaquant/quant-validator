"""quant-validator: Rigorous validation toolkit for quantitative trading."""

__version__ = "0.3.0"

from .core import (
    Metrics,
    Simulator,
    PlaceboTest,
    LOAO,
    WalkForward,
    DSR,
    Validator,
    build_events,
    simulate_account,
)

__all__ = [
    "Metrics",
    "Simulator",
    "PlaceboTest",
    "LOAO",
    "WalkForward",
    "DSR",
    "Validator",
    "build_events",
    "simulate_account",
]
