
"""quant-validator: Rigorous validation toolkit."""

import numpy as np
import pandas as pd
from scipy import stats
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Callable
import warnings
warnings.filterwarnings("ignore")


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
        dd = (equity / peak) - 1
        return dd.min()

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
            "n_nonzero": (r != 0).sum(),
            "fraction_active": (r != 0).mean() if len(r) > 0 else np.nan,
            "mean_bps": r.mean() * 1e4,
            "sharpe_calendar": sharpe_cal,
            "sharpe_active": sharpe_act,
            "sharpe_ratio": sharpe_act / sharpe_cal if sharpe_cal else np.nan,
            "max_dd": Metrics.max_dd(equity),
            "pf": Metrics.pf(r),
            "hit_rate": (r[r != 0] > 0).mean() if (r != 0).any() else np.nan,
        }


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
            "placebo_p95": np.percentile(pfs, 95),
            "percentile": (pfs < real_pf).mean() * 100,
            "p_value": (pfs >= real_pf).mean(),
        }


class LOAO:
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
            daily = pd.Series(eq["equity"].values, index=eq_ts).resample("D").last().ffill()
            ret = daily.pct_change().dropna()
            results.append({
                "held_out": held, "n_trades": len(sub),
                "sharpe": Metrics.sharpe(ret, periods_per_year, "calendar"),
            })
        return pd.DataFrame(results)


class WalkForward:
    def __init__(self, simulator_fn, periods_per_year=365.0, **kwargs):
        self.simulator_fn = simulator_fn
        self.ppy = periods_per_year
        self.kwargs = kwargs
    def _sharpe(self, sub):
        eq = self.simulator_fn(sub, **self.kwargs)
        if eq is None or len(eq) < 2:
            return np.nan
        eq_ts = pd.to_datetime(eq["ts"], utc=True)
        daily = pd.Series(eq["equity"].values, index=eq_ts).resample("D").last().ffill()
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


class DSR:
    @staticmethod
    def compute(returns, n_trials=1, periods_per_year=365.0, base="calendar"):
        r = returns.dropna()
        if base == "active":
            r = r[r != 0]
        if len(r) < 30:
            return {"error": f"too few observations ({len(r)})"}
        sr_d = r.mean() / r.std(ddof=1)
        sr_a = sr_d * np.sqrt(periods_per_year)
        skew = r.skew()
        kurt = r.kurtosis() + 3
        log_n = np.log(max(n_trials, 2))
        e_max = np.sqrt(2 * log_n) - (np.log(log_n) + np.log(4 * np.pi)) / (2 * np.sqrt(2 * log_n))
        se = np.sqrt((1 - skew * sr_d + (kurt - 1) / 4 * sr_d ** 2) / (len(r) - 1))
        z = (sr_d - e_max) / se
        dsr = stats.norm.cdf(z)
        return {
            "sharpe_annual": sr_a, "n_obs": len(r), "n_trials": n_trials,
            "skew": skew, "kurt": kurt, "e_max_sharpe": e_max,
            "dsr": dsr, "significant": dsr > 0.95,
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
        self.results["loao"] = LOAO(self._sim).run(self.trades, self.asset_col, self.periods_per_year)
        self.results["walkforward"] = WalkForward(self._sim, self.periods_per_year).split_analysis(
            self.trades, split_date, self.entry_col)
        self.results["dsr"] = DSR.compute(ret, n_trials, self.periods_per_year)

    def summary(self):
        lines = ["=" * 70, " QUANT-VALIDATOR SUMMARY", "=" * 70]
        if self._skipped:
            lines.append("\n--- EXECUTION ---")
            lines.append(f"  Trades generated:  {self._skipped.get('total_generated', 0):,}")
            lines.append(f"  Trades executed:   {self._skipped.get('executed', 0):,}")
            lines.append(f"  Execution rate:    {self._skipped.get('execution_rate', 0):.1%}")
        if "basic" in self.results:
            b = self.results["basic"]
            lines.append("\n--- BASIC METRICS ---")
            lines.append(f"  Sharpe (calendar): {b.get('sharpe_calendar', 0):.3f}")
            lines.append(f"  Sharpe (active):   {b.get('sharpe_active', 0):.3f}")
            lines.append(f"  Max DD:            {b.get('max_dd', 0):.2%}")
            lines.append(f"  PF:                {b.get('pf', 0):.3f}")
        if "dsr" in self.results:
            d = self.results["dsr"]
            lines.append("\n--- DSR ---")
            lines.append(f"  DSR:               {d.get('dsr', 0):.4f}")
        lines.append("=" * 70)
        return "\n".join(lines)
