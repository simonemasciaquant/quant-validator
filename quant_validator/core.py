"""quant-validator: Rigorous validation toolkit. Version 0.2.0."""

import numpy as np
import pandas as pd
from scipy import stats
from dataclasses import dataclass, field
from typing import Optional, Dict
import warnings
warnings.filterwarnings("ignore")

EULER_GAMMA = 0.5772156649015329


class Metrics:
    @staticmethod
    def sharpe(returns, periods_per_year=365.0, base="calendar"):
        r = returns.dropna()
        if base == "active":
            r = r[r != 0]
        if len(r) < 2 or r.std(ddof=1) == 0:
            return np.nan
        return (r.mean() / r.std(ddof=1)) * np.sqrt(periods_per_year)

    @staticmethod
    def max_dd(equity):
        if len(equity) < 2:
            return np.nan
        peak = equity.cummax()
        return (equity / peak - 1).min()

    @staticmethod
    def pf(returns):
        r = returns.dropna()
        pos = r[r > 0].sum()
        neg = abs(r[r < 0].sum())
        return pos / neg if neg > 0 else np.inf

    @staticmethod
    def all_metrics(returns, equity=None, periods_per_year=365.0):
        r = returns.dropna()
        if equity is None:
            equity = (1 + r).cumprod()
        sharpe_cal = Metrics.sharpe(r, periods_per_year, "calendar")
        sharpe_act = Metrics.sharpe(r, periods_per_year, "active")
        return {
            "n": len(r),
            "sharpe_calendar": sharpe_cal,
            "sharpe_active": sharpe_act,
            "sharpe_ratio": sharpe_act / sharpe_cal if sharpe_cal else np.nan,
            "max_dd": Metrics.max_dd(equity),
            "pf": Metrics.pf(r),
        }



    @staticmethod
    def sharpe_from_equity(equity_df, periods_per_year=365):
        """Calendar Sharpe from an equity DataFrame (columns: ts, equity)."""
        eq_ts = pd.to_datetime(equity_df["ts"], utc=True).dt.tz_localize(None)
        daily = (
            pd.Series(equity_df["equity"].values, index=eq_ts)
            .resample("D").last().ffill()
        )
        r = daily.pct_change().dropna()
        if len(r) < 2 or r.std(ddof=1) == 0:
            return float("nan")
        return float(r.mean() / r.std(ddof=1) * np.sqrt(periods_per_year))

    @staticmethod
    def cagr_from_equity(equity_df, capital_start):
        """CAGR from an equity DataFrame."""
        eq_ts = pd.to_datetime(equity_df["ts"], utc=True).dt.tz_localize(None)
        years = (eq_ts.iloc[-1] - eq_ts.iloc[0]).days / 365.25
        total = equity_df["equity"].iloc[-1] / capital_start - 1
        if years <= 0:
            return float("nan")
        return float((1 + total) ** (1 / years) - 1)

    @staticmethod
    def max_dd_from_equity(equity_df):
        """Max DD from an equity DataFrame (daily resampled)."""
        eq_ts = pd.to_datetime(equity_df["ts"], utc=True).dt.tz_localize(None)
        daily = (
            pd.Series(equity_df["equity"].values, index=eq_ts)
            .resample("D").last().ffill()
        )
        eq = daily.values
        peak = np.maximum.accumulate(eq)
        dd = (eq - peak) / peak
        return float(dd.min())


class Simulator:
    @staticmethod
    def flat_sizing(trades, entry_col="entry_ts", exit_col="exit_ts",
                    ret_col="ret_net", capital=10000, max_conc=10,
                    asset_col="asset", return_log=False):
        trades = trades.sort_values(entry_col).reset_index(drop=True).copy()
        if len(trades) == 0:
            empty = pd.DataFrame(columns=["ts", "equity"])
            return (empty, pd.DataFrame(), {"executed": 0}) if return_log else empty
        cap = capital
        eq = [(trades.iloc[0][entry_col], cap)]
        open_pos = []
        log = []
        skipped = {"max_concurrent": 0, "already_open": 0, "executed": 0}
        for _, row in trades.iterrows():
            now = row[entry_col]
            open_pos.sort(key=lambda p: p["exit_ts"])
            while open_pos and open_pos[0]["exit_ts"] <= now:
                p = open_pos.pop(0)
                cap += p["pnl"]
                eq.append((p["exit_ts"], cap))
                log.append({**p, "capital_after": cap})
            if len(open_pos) >= max_conc:
                skipped["max_concurrent"] += 1
                continue
            if any(p["asset"] == row[asset_col] for p in open_pos):
                skipped["already_open"] += 1
                continue
            size = cap / max_conc
            pnl = size * row[ret_col]
            open_pos.append({
                "asset": row[asset_col], "entry_ts": row[entry_col],
                "exit_ts": row[exit_col], "size": size,
                "ret_net": row[ret_col], "pnl": pnl,
            })
            skipped["executed"] += 1
        for p in open_pos:
            cap += p["pnl"]
            eq.append((p["exit_ts"], cap))
            log.append({**p, "capital_after": cap})
        eq_df = pd.DataFrame(eq, columns=["ts", "equity"]).sort_values("ts").reset_index(drop=True)
        if return_log:
            skipped["total_generated"] = len(trades)
            skipped["execution_rate"] = skipped["executed"] / len(trades) if len(trades) > 0 else 0
            return eq_df, pd.DataFrame(log), skipped
        return eq_df


class PlaceboTest:
    def __init__(self, n_iter=300, random_state=42):
        self.n_iter = n_iter
        self.rng = np.random.default_rng(random_state)
    def run(self, returns):
        r = returns.dropna().values
        if len(r) < 30:
            return {"error": f"too few trades ({len(r)})"}
        real_pf = r[r > 0].sum() / abs(r[r < 0].sum()) if r[r < 0].sum() != 0 else np.inf
        pfs = []
        for _ in range(self.n_iter):
            flip = self.rng.random(len(r)) < 0.5
            r_p = r.copy()
            r_p[flip] = -r_p[flip]
            pos = r_p[r_p > 0].sum()
            neg = abs(r_p[r_p < 0].sum())
            pfs.append(pos / neg if neg > 0 else np.inf)
        pfs = np.array(pfs)
        return {
            "real_pf": real_pf,
            "placebo_mean": pfs.mean(),
            "percentile": (pfs < real_pf).mean() * 100,
        }


class DSR:
    """Deflated Sharpe Ratio - versione corretta."""

    @staticmethod
    def expected_max_sharpe(n_trials, sr_var):
        if n_trials < 2:
            return 0.0
        z1 = stats.norm.ppf(1.0 - 1.0 / n_trials)
        z2 = stats.norm.ppf(1.0 - 1.0 / (n_trials * np.e))
        return float(np.sqrt(sr_var) * ((1.0 - EULER_GAMMA) * z1 + EULER_GAMMA * z2))

    @staticmethod
    def compute(returns, n_trials=1, trial_sharpes=None, periods_per_year=365):
        r = np.asarray(returns, dtype=float)
        r = r[~np.isnan(r)]
        T = len(r)
        if T < 30:
            return {"error": f"Servono almeno 30 osservazioni (T={T})"}
        sr = r.mean() / r.std(ddof=1)
        skew = stats.skew(r)
        kurt = stats.kurtosis(r, fisher=False)
        var_term = max(1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr ** 2, 1e-12)
        se = np.sqrt(var_term / (T - 1))
        if trial_sharpes is not None:
            ts = np.asarray(trial_sharpes, dtype=float)
            ts = ts[~np.isnan(ts)]
            n_trials = len(ts)
            sr_var = float(ts.var(ddof=1)) if n_trials > 1 else 0.0
            var_src = "trial_sharpes"
        else:
            sr_var = 1.0 / (T - 1)
            var_src = "nulla 1/(T-1)"
        sr0 = DSR.expected_max_sharpe(n_trials, sr_var)
        ann = np.sqrt(periods_per_year)
        dsr = float(stats.norm.cdf((sr - sr0) / se))
        return {
            "n_obs": T,
            "n_trials": int(n_trials),
            "sharpe_annual": sr * ann,
            "sr0_annual": sr0 * ann,
            "skew": float(skew),
            "kurtosis": float(kurt),
            "psr": float(stats.norm.cdf(sr / se)),
            "dsr": dsr,
            "var_source": var_src,
            "significant": dsr > 0.95,
        }


@dataclass
class Validator:
    trades: pd.DataFrame
    asset_col: str = "asset"
    entry_col: str = "entry_ts"
    exit_col: str = "exit_ts"
    ret_col: str = "ret_net"
    capital: float = 10000
    max_conc: int = 10
    periods_per_year: float = 365.0

    results: Dict = field(default_factory=dict, init=False)
    _equity: Optional[pd.DataFrame] = field(default=None, init=False)
    _skipped: Optional[Dict] = field(default=None, init=False)

    def __post_init__(self):
        self.trades = self.trades.copy()
        self.trades[self.entry_col] = pd.to_datetime(self.trades[self.entry_col], utc=True)
        self.trades[self.exit_col] = pd.to_datetime(self.trades[self.exit_col], utc=True)
        self.trades = self.trades.sort_values(self.entry_col).reset_index(drop=True)

    def _sim(self, trades):
        return Simulator.flat_sizing(
            trades, entry_col=self.entry_col, exit_col=self.exit_col,
            ret_col=self.ret_col, capital=self.capital,
            max_conc=self.max_conc, asset_col=self.asset_col)

    def _rets(self, eq):
        if eq is None or len(eq) < 2:
            return pd.Series(dtype=float)
        eq_ts = pd.to_datetime(eq["ts"], utc=True)
        daily = pd.Series(eq["equity"].values, index=eq_ts).resample("D").last().ffill()
        return daily.pct_change().dropna()

    def run_all(self, split_date="2024-01-01", n_trials=1):
        self._equity, _, self._skipped = Simulator.flat_sizing(
            self.trades, entry_col=self.entry_col, exit_col=self.exit_col,
            ret_col=self.ret_col, capital=self.capital,
            max_conc=self.max_conc, asset_col=self.asset_col, return_log=True)
        ret = self._rets(self._equity)
        equity = pd.Series(self._equity["equity"].values,
                           index=pd.to_datetime(self._equity["ts"], utc=True))
        self.results["basic"] = Metrics.all_metrics(ret, equity, self.periods_per_year)
        self.results["placebo"] = PlaceboTest().run(self.trades[self.ret_col])
        self.results["dsr"] = DSR.compute(ret, n_trials=n_trials, periods_per_year=int(self.periods_per_year))

    def summary(self):
        lines = ["=" * 70, " QUANT-VALIDATOR SUMMARY", "=" * 70]
        if "basic" in self.results:
            b = self.results["basic"]
            lines.append(f"  Sharpe (calendar): {b.get('sharpe_calendar', 0):.3f}")
            lines.append(f"  Sharpe (active):   {b.get('sharpe_active', 0):.3f}")
            lines.append(f"  Max DD:            {b.get('max_dd', 0):.2%}")
        if "dsr" in self.results:
            d = self.results["dsr"]
            if "error" not in d:
                lines.append(f"  Sharpe (annual):   {d['sharpe_annual']:.3f}")
                lines.append(f"  DSR:               {d['dsr']:.4f}")
        lines.append("=" * 70)
        return "\n".join(lines)



# ============================================================
# v0.3.0 — Event clustering & account simulation
# ============================================================

def build_events(trades_df, gap_minutes=30):
    """
    Assign an event_id to each trade, grouping trades into temporal
    clusters.

    Two consecutive trades belong to the same cluster if the gap between
    their entry_ts is <= gap_minutes. Otherwise a new cluster starts.

    Parameters
    ----------
    trades_df : pd.DataFrame
        Must contain column ``entry_ts`` (datetime64 UTC, tz-aware).
    gap_minutes : int, default 30
        Maximum gap (in minutes) between consecutive trades to be
        considered part of the same event.

    Returns
    -------
    pd.DataFrame
        Same as input, sorted by ``entry_ts``, with an added integer
        column ``event_id`` starting at 0.
    """
    t = trades_df.sort_values("entry_ts").reset_index(drop=True).copy()

    if len(t) == 0:
        t["event_id"] = pd.Series(dtype="int64")
        return t

    event_id = 0
    t.loc[0, "event_id"] = 0
    for i in range(1, len(t)):
        prev_ts = t.iloc[i - 1]["entry_ts"]
        curr_ts = t.iloc[i]["entry_ts"]
        gap = (curr_ts - prev_ts).total_seconds() / 60.0
        if gap > gap_minutes:
            event_id += 1
        t.loc[i, "event_id"] = event_id

    t["event_id"] = t["event_id"].astype("int64")
    return t


def simulate_account(trades_df, capital_start=10000.0, max_concurrent=10,
                     assets_order=None):
    """
    Simulate a compounding-size account with a chronological cap.

    At each trade entry:
      - close any open position whose ``exit_ts`` <= current entry_ts
        (in exit_ts order),
      - skip if ``max_concurrent`` positions are already open,
      - skip if the same asset is already held,
      - otherwise open with size = current_capital / max_concurrent.

    Returns
    -------
    executed_df : pd.DataFrame
        Columns: entry_ts, exit_ts, asset, ret_net, size, pnl.
    equity_df : pd.DataFrame
        Columns: ts, equity. Sorted by ts.
    final_capital : float
    """
    if assets_order is None:
        assets_order = sorted(trades_df["asset"].unique())

    asset_rank = {a: i for i, a in enumerate(assets_order)}

    t = trades_df.copy()
    t["_asset_rank"] = t["asset"].map(asset_rank)
    t = t.sort_values(["entry_ts", "_asset_rank"]).reset_index(drop=True)
    t = t.drop(columns=["_asset_rank"])

    if len(t) == 0:
        empty_exec = pd.DataFrame(
            columns=["entry_ts", "exit_ts", "asset", "ret_net", "size", "pnl"]
        )
        empty_eq = pd.DataFrame(columns=["ts", "equity"])
        return empty_exec, empty_eq, float(capital_start)

    capital = float(capital_start)
    open_pos = []
    executed = []
    equity = [(t.iloc[0]["entry_ts"], capital)]

    for _, row in t.iterrows():
        now = row["entry_ts"]

        open_pos.sort(key=lambda p: p["exit_ts"])
        while open_pos and open_pos[0]["exit_ts"] <= now:
            p = open_pos.pop(0)
            capital += p["pnl"]
            equity.append((p["exit_ts"], capital))

        if len(open_pos) >= max_concurrent:
            continue

        if any(p["asset"] == row["asset"] for p in open_pos):
            continue

        size = capital / max_concurrent
        pnl = size * row["ret_net"]

        open_pos.append({
            "asset": row["asset"],
            "entry_ts": now,
            "exit_ts": row["exit_ts"],
            "size": size,
            "ret_net": row["ret_net"],
            "pnl": pnl,
        })
        executed.append({
            "entry_ts": now,
            "exit_ts": row["exit_ts"],
            "asset": row["asset"],
            "ret_net": row["ret_net"],
            "size": size,
            "pnl": pnl,
        })

    # FIX v0.3.0: close residual positions in exit_ts order
    open_pos.sort(key=lambda p: p["exit_ts"])
    for p in open_pos:
        capital += p["pnl"]
        equity.append((p["exit_ts"], capital))

    executed_df = pd.DataFrame(executed).reset_index(drop=True)
    equity_df = (
        pd.DataFrame(equity, columns=["ts", "equity"])
        .sort_values("ts")
        .reset_index(drop=True)
    )

    return executed_df, equity_df, float(capital)



# ============================================================
# v0.3.0 — LOAO & WalkForward
# ============================================================

class LOAO:
    """Leave-One-Asset-Out cross-validation."""

    def __init__(self, simulator_fn, **kwargs):
        self.simulator_fn = simulator_fn
        self.kwargs = kwargs

    def run(self, trades, asset_col="asset", periods_per_year=365.0):
        results = []
        for held in trades[asset_col].unique():
            sub = trades[trades[asset_col] != held]
            if len(sub) < 50:
                continue
            eq = self.simulator_fn(sub, **self.kwargs)
            if eq is None or len(eq) < 2:
                continue
            eq_ts = pd.to_datetime(eq["ts"], utc=True)
            daily = (
                pd.Series(eq["equity"].values, index=eq_ts)
                .resample("D").last().ffill()
            )
            ret = daily.pct_change().dropna()
            results.append({
                "held_out": held,
                "n_trades": len(sub),
                "sharpe": Metrics.sharpe(ret, periods_per_year, "calendar"),
            })
        return pd.DataFrame(results)


class WalkForward:
    """In-sample / out-of-sample split analysis."""

    def __init__(self, simulator_fn, periods_per_year=365.0, **kwargs):
        self.simulator_fn = simulator_fn
        self.ppy = periods_per_year
        self.kwargs = kwargs

    def _sharpe(self, sub):
        eq = self.simulator_fn(sub, **self.kwargs)
        if eq is None or len(eq) < 2:
            return np.nan
        eq_ts = pd.to_datetime(eq["ts"], utc=True)
        daily = (
            pd.Series(eq["equity"].values, index=eq_ts)
            .resample("D").last().ffill()
        )
        ret = daily.pct_change().dropna()
        return Metrics.sharpe(ret, self.ppy, "calendar")

    def split_analysis(self, trades, split_date, entry_col="entry_ts"):
        trades = trades.copy()
        trades[entry_col] = pd.to_datetime(trades[entry_col], utc=True)
        split = pd.Timestamp(split_date, tz="UTC")
        is_t = trades[trades[entry_col] < split]
        oos_t = trades[trades[entry_col] >= split]
        is_s = self._sharpe(is_t)
        oos_s = self._sharpe(oos_t)
        return {
            "is_n": len(is_t), "oos_n": len(oos_t),
            "is_sharpe": is_s, "oos_sharpe": oos_s,
            "stability_ratio": oos_s / is_s if is_s else np.nan,
        }
