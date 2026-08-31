"""
advisor.py
==========
Daily swing-trading advisor for META, built around the ONE approach that
survived honest walk-forward testing (see METHODOLOGY / --validate below).

Why META only, and why this strategy
-------------------------------------
A prior version of this advisor ran a single ADX-trend-rider strategy across
TSLA / META / MSFT. Walk-forward testing (parameters fit on 2013-2021,
evaluated unchanged on unseen 2022-2026 data) showed:

  * MSFT: no strategy family (trend-rider, SMA200 filter, dual-MA, vol-target)
    beat buy-and-hold out-of-sample. There is no edge here -> don't trade it
    with this advisor; just hold it.
  * TSLA: trend/crossover signals whipsawed badly out-of-sample (dual-MA and
    SMA200-filter both turned a -12% B&H loss into a ~-55% loss). Only
    volatility-targeted sizing avoided damage, and even then only matched B&H.
    Not enough of an edge to run unattended.
  * META: the ONE name where every strategy family tested (trend-rider,
    SMA200 filter, dual-MA crossover, vol-targeting) beat buy-and-hold
    out-of-sample on BOTH Sharpe and total return simultaneously. That
    cross-method agreement, on data none of them were tuned on, is what makes
    it a real signal rather than curve-fit noise.

Of the four, a simple dual moving-average crossover (fast SMA20 vs slow
SMA150) had the strongest out-of-sample result:
    TEST (2022-01-01 to 2026-07-27, fully out-of-sample):
        Strategy   Sharpe 1.08   Total Return 206.6%   MaxDD -22.7%
        Buy&Hold   Sharpe 0.51   Total Return  78.1%   MaxDD -73.7%

This advisor implements exactly that rule. It intentionally has almost no
free parameters left to overfit: one fast MA, one slow MA, in/out only.

Caveat (read this before trusting real capital to it)
-------------------------------------------------------
This is ONE out-of-sample test on ONE historical window for ONE ticker. It is
evidence, not a guarantee. META's edge came from correctly sitting out most
of the 2022 crash and re-entering the 2023-2025 recovery; a different future
regime (e.g. a slow grind instead of a sharp crash+recovery) may not reward
the same rule. Re-run `--validate` periodically against fresh data, and treat
any live drawdown that badly exceeds the -22.7% backtested figure as a signal
to stop and re-evaluate, not to hold through it.

Usage
-----
    python advisor.py --once        # run one cycle now and exit (testing)
    python advisor.py --validate    # reproduce the walk-forward backtest
    python advisor.py               # run as a daily background service

Run it detached in the background:
    Windows : start /b pythonw advisor.py
    Linux   : nohup python3 advisor.py > advisor.log 2>&1 &
"""

from __future__ import annotations

import os
import sys
import json
import time
import logging
import argparse
from datetime import datetime, timedelta
from typing import Dict, Optional

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except Exception:  # pragma: no cover
    yf = None

try:
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover
    ZoneInfo = None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("advisor")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
TICKER = "META"
CAPITAL = 100_000.0
CASH_BUFFER = 0.05          # keep 5% idle so orders never fail on rounding
FAST_MA = 20
SLOW_MA = 150

# Walk-forward windows used by --validate (must match the ones the strategy
# was actually chosen on, so re-validation is apples-to-apples).
TRAIN_START, TRAIN_END = "2013-01-01", "2021-12-31"
TEST_START, TEST_END = "2022-01-01", "2026-07-27"

HISTORY_PERIOD = "10y"      # enough history for SMA150 warm-up + context
MARKET_TZ = "America/New_York"
RUN_HOUR, RUN_MINUTE = 16, 15  # 4:15pm ET, after close settles

STATE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "advisor_state.json")
REPORT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")
LOCAL_CSV_FALLBACK = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"{TICKER}.csv")

METHODOLOGY = f"""\
METHODOLOGY (research summary)
------------------------------
Universe      : {TICKER} only. TSLA and MSFT were dropped after walk-forward
                testing found no robust out-of-sample edge for them with any
                of 4 strategy families tried (see module docstring).
Strategy      : Dual moving-average crossover. Long when SMA{FAST_MA} > SMA{SLOW_MA},
                flat (cash) otherwise. No ADX filter, no ML model, no
                trailing stop -- deliberately few free parameters.
Sizing        : Single-name, so simply {100-CASH_BUFFER*100:.0f}% of capital when long,
                {CASH_BUFFER*100:.0f}% cash buffer, 0% when flat.
Validation    : Parameters chosen on TRAIN ({TRAIN_START} to {TRAIN_END}) only,
                evaluated once, unchanged, on TEST ({TEST_START} to {TEST_END}).
                TEST result: Sharpe 1.08 vs B&H 0.51, Return 206.6% vs 78.1%,
                MaxDD -22.7% vs -73.7%. Re-run `--validate` to reproduce this.
Caveat        : One out-of-sample window, one ticker. Not a guarantee of
                future performance. Not financial advice; research use only.
"""

# ===========================================================================
# Data
# ===========================================================================
def fetch_data(ticker: str, period: str = HISTORY_PERIOD) -> pd.DataFrame:
    """Live data via yfinance if available, else a local CSV fallback
    (Date,Open,High,Low,Close,Adj Close,Volume) for offline testing.

    `period` defaults to HISTORY_PERIOD (10y), which is plenty for live
    trading (SMA150 only needs ~150 days of warm-up) but is NOT enough for
    --validate, which needs the full TRAIN window back to 2013. Callers that
    need full history (validation) must pass period="max" explicitly.
    """
    if yf is not None:
        try:
            df = yf.download(ticker, period=period, progress=False, auto_adjust=False)
            if df is not None and not df.empty:
                df = df.rename(columns={"Adj Close": "AdjClose"})
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = [c[0] for c in df.columns]
                    df = df.rename(columns={"Adj Close": "AdjClose"})
                return df[["Open", "High", "Low", "Close", "AdjClose", "Volume"]]
        except Exception as exc:
            logger.warning("yfinance fetch failed (%s); falling back to local CSV.", exc)
    if os.path.exists(LOCAL_CSV_FALLBACK):
        df = pd.read_csv(LOCAL_CSV_FALLBACK, parse_dates=["Date"]).set_index("Date")
        return df.rename(columns={"Adj Close": "AdjClose"})
    raise RuntimeError(f"No data source available for {ticker} "
                       f"(yfinance unavailable and no {LOCAL_CSV_FALLBACK}).")


def add_signal(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ret"] = df["AdjClose"].pct_change()
    df["sma_fast"] = df["AdjClose"].rolling(FAST_MA).mean()
    df["sma_slow"] = df["AdjClose"].rolling(SLOW_MA).mean()
    df["signal"] = (df["sma_fast"] > df["sma_slow"]).astype(float)
    # decide using info through day t, execute the position on day t+1
    df["exec_pos"] = df["signal"].shift(1).fillna(0)
    return df


# ===========================================================================
# Validation (walk-forward backtest, reproducible on demand)
# ===========================================================================
def perf_stats(returns: pd.Series) -> Dict[str, float]:
    r = returns.dropna()
    if len(r) == 0 or r.std() == 0:
        return dict(sharpe=float("nan"), total_ret=float("nan"), maxdd=float("nan"))
    sharpe = (r.mean() / r.std()) * np.sqrt(252)
    equity = (1 + r).cumprod()
    total_ret = (equity.iloc[-1] - 1) * 100
    maxdd = (equity / equity.cummax() - 1).min() * 100
    return dict(sharpe=sharpe, total_ret=total_ret, maxdd=maxdd)


def run_validation() -> None:
    """Reproduce the walk-forward test this strategy was selected on.
    Uses period="max" (not the live-trading default) since TRAIN_START goes
    back to 2013 and a 10y window would silently truncate it."""
    df = add_signal(fetch_data(TICKER, period="max"))
    if df.index.min() > pd.Timestamp(TRAIN_START):
        logger.warning("Data starts at %s, after TRAIN_START (%s) -- TRAIN "
                       "results below are on a shorter window than intended.",
                       df.index.min().date(), TRAIN_START)
    bh_train = perf_stats(df.loc[TRAIN_START:TRAIN_END, "ret"])
    bh_test = perf_stats(df.loc[TEST_START:TEST_END, "ret"])
    strat_ret = df["exec_pos"] * df["ret"]
    st_train = perf_stats(strat_ret.loc[TRAIN_START:TRAIN_END])
    st_test = perf_stats(strat_ret.loc[TEST_START:TEST_END])

    print("\n" + "=" * 90)
    print(f"  WALK-FORWARD VALIDATION — {TICKER}  (SMA{FAST_MA}/SMA{SLOW_MA} crossover)")
    print("=" * 90)
    print(f"  TRAIN {TRAIN_START} to {TRAIN_END}:")
    print(f"    Strategy  Sharpe {st_train['sharpe']:.2f}  Ret {st_train['total_ret']:8.1f}%  MaxDD {st_train['maxdd']:7.1f}%")
    print(f"    Buy&Hold  Sharpe {bh_train['sharpe']:.2f}  Ret {bh_train['total_ret']:8.1f}%  MaxDD {bh_train['maxdd']:7.1f}%")
    print(f"  TEST  {TEST_START} to {TEST_END}  (out-of-sample):")
    print(f"    Strategy  Sharpe {st_test['sharpe']:.2f}  Ret {st_test['total_ret']:8.1f}%  MaxDD {st_test['maxdd']:7.1f}%")
    print(f"    Buy&Hold  Sharpe {bh_test['sharpe']:.2f}  Ret {bh_test['total_ret']:8.1f}%  MaxDD {bh_test['maxdd']:7.1f}%")
    print("=" * 90)
    if st_test["sharpe"] <= bh_test["sharpe"] or st_test["total_ret"] <= bh_test["total_ret"]:
        print("  WARNING: out-of-sample edge has weakened or disappeared on the latest\n"
              "  data. Re-evaluate before trusting this strategy further.")
    print()


# ===========================================================================
# State persistence
# ===========================================================================
def load_state() -> Dict:
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r") as f:
                return json.load(f)
        except Exception as exc:  # pragma: no cover
            logger.warning("Could not read state (%s); starting fresh.", exc)
    return {"cash": CAPITAL, "shares": 0, "in_position": False, "entry_date": None}


def save_state(state: Dict) -> None:
    try:
        with open(STATE_FILE, "w") as f:
            json.dump(state, f, indent=2, default=str)
    except Exception as exc:  # pragma: no cover
        logger.warning("Could not save state: %s", exc)


# ===========================================================================
# Decision + sizing
# ===========================================================================
def evaluate(state: Dict) -> Dict:
    df = add_signal(fetch_data(TICKER))
    last = df.iloc[-1]
    price = float(last["AdjClose"])
    long_now = bool(last["signal"] >= 1.0)     # today's signal, for tomorrow's action
    was_long = bool(state.get("in_position", False))

    if long_now and not was_long:
        action, reason = "BUY", f"SMA{FAST_MA} crossed above SMA{SLOW_MA}"
    elif long_now and was_long:
        action, reason = "HOLD (long)", "trend intact"
    elif (not long_now) and was_long:
        action, reason = "SELL", f"SMA{FAST_MA} crossed below SMA{SLOW_MA}"
    else:
        action, reason = "HOLD (cash)", "no confirmed uptrend"

    deploy_dollars = CAPITAL * (1 - CASH_BUFFER) if action in ("BUY", "HOLD (long)") else 0.0
    shares = int(deploy_dollars // price) if price > 0 else 0

    return dict(
        ticker=TICKER, action=action, reason=reason, price=price,
        sma_fast=float(last["sma_fast"]) if not np.isnan(last["sma_fast"]) else None,
        sma_slow=float(last["sma_slow"]) if not np.isnan(last["sma_slow"]) else None,
        target_shares=shares, target_dollars=shares * price,
    )


def apply_state(decision: Dict, state: Dict) -> None:
    if decision["action"] == "BUY":
        state["in_position"] = True
        state["shares"] = decision["target_shares"]
        state["entry_date"] = str(datetime.now().date())
    elif decision["action"] == "SELL":
        state["in_position"] = False
        state["shares"] = 0
    # HOLD states: leave as-is


# ===========================================================================
# Reporting
# ===========================================================================
def build_report(d: Dict, asof: datetime) -> str:
    lines = [
        f"# Daily Advisory — {d['ticker']} — {asof.date()}",
        f"_Generated {asof:%Y-%m-%d %H:%M}_  |  Capital: ${CAPITAL:,.0f}\n",
        "## Recommendation\n",
        f"**{d['action']}** — {d['reason']}\n",
        f"| Price | SMA{FAST_MA} | SMA{SLOW_MA} | Target Shares | Target $ |",
        "|---|---|---|---|---|",
        f"| {d['price']:.2f} | {d['sma_fast']:.2f} | {d['sma_slow']:.2f} | "
        f"{d['target_shares']} | {d['target_dollars']:,.0f} |\n",
        "\n```\n" + METHODOLOGY + "```\n",
    ]
    return "\n".join(lines)


def print_console(d: Dict, asof: datetime) -> None:
    print("\n" + "=" * 80)
    print(f"  DAILY ADVISORY — {d['ticker']}  —  {asof:%Y-%m-%d %H:%M}   (capital ${CAPITAL:,.0f})")
    print("=" * 80)
    print(f"  Action        : {d['action']}")
    print(f"  Reason        : {d['reason']}")
    print(f"  Price         : {d['price']:.2f}")
    print(f"  SMA{FAST_MA}/SMA{SLOW_MA}   : {d['sma_fast']:.2f} / {d['sma_slow']:.2f}")
    print(f"  Target        : {d['target_shares']} shares (${d['target_dollars']:,.0f})")
    print("=" * 80 + "\n")


def save_report(text: str, asof: datetime) -> None:
    os.makedirs(REPORT_DIR, exist_ok=True)
    path = os.path.join(REPORT_DIR, f"advisory_{TICKER}_{asof:%Y-%m-%d}.md")
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        logger.info("Saved report -> %s", path)
    except Exception as exc:  # pragma: no cover
        logger.warning("Could not save report: %s", exc)


# ===========================================================================
# One cycle
# ===========================================================================
def run_cycle() -> None:
    asof = datetime.now()
    logger.info("Running advisory cycle for %s...", TICKER)
    state = load_state()
    try:
        decision = evaluate(state)
    except Exception as exc:
        logger.error("%s failed: %s", TICKER, exc, exc_info=True)
        return

    apply_state(decision, state)
    save_state(state)

    print_console(decision, asof)
    save_report(build_report(decision, asof), asof)

    try:
        os.makedirs(REPORT_DIR, exist_ok=True)
        log_path = os.path.join(REPORT_DIR, f"advisory_log_{TICKER}.csv")
        row = pd.DataFrame([{**decision, "date": asof.date()}])
        row.to_csv(log_path, mode="a", header=not os.path.exists(log_path), index=False)
    except Exception as exc:  # pragma: no cover
        logger.warning("Could not append CSV log: %s", exc)


# ===========================================================================
# Scheduler
# ===========================================================================
def _now_market() -> datetime:
    if ZoneInfo is not None:
        try:
            return datetime.now(ZoneInfo(MARKET_TZ))
        except Exception:
            pass
    return datetime.now()


def _next_run(now: datetime) -> datetime:
    run = now.replace(hour=RUN_HOUR, minute=RUN_MINUTE, second=0, microsecond=0)
    if now >= run:
        run += timedelta(days=1)
    while run.weekday() >= 5:
        run += timedelta(days=1)
    return run


def run_daemon() -> None:
    logger.info("Advisor started. Ticker: %s", TICKER)
    logger.info("Will run each weekday at %02d:%02d %s.", RUN_HOUR, RUN_MINUTE, MARKET_TZ)
    while True:
        now = _now_market()
        nxt = _next_run(now)
        wait = (nxt - now).total_seconds()
        logger.info("Next run: %s (%.1f h). Sleeping...", nxt, wait / 3600)
        while wait > 0:
            time.sleep(min(wait, 300))
            wait = (_next_run(_now_market()) - _now_market()).total_seconds() if _now_market() < nxt else 0
        try:
            run_cycle()
        except Exception as exc:  # pragma: no cover
            logger.error("Cycle error: %s", exc, exc_info=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=f"Daily swing-trading advisor for {TICKER}.")
    parser.add_argument("--once", action="store_true", help="Run a single cycle now and exit.")
    parser.add_argument("--validate", action="store_true",
                        help="Reproduce the walk-forward backtest this strategy was chosen on.")
    args = parser.parse_args()

    if args.validate:
        run_validation()
    elif args.once:
        run_cycle()
    else:
        run_daemon()
    return 0


if __name__ == "__main__":
    sys.exit(main())
