# Trading-Algorithm

A small Python research and paper-trading advisor for META built around a dual moving-average crossover. The project intentionally keeps the strategy simple so the walk-forward test is easy to reproduce and audit.

## What it does

- Downloads historical META prices with yfinance (or uses the checked-in META.csv fallback).
- Computes a 20-day and 150-day simple moving average.
- Enters a long position when SMA20 is above SMA150 and stays in cash otherwise.
- Produces a daily recommendation, position sizing, and a text report.
- Supports an out-of-sample walk-forward validation run.

## Files

- advisor.py contains data loading, signal generation, validation, position sizing, reporting, and the daily scheduler.
- META.csv is the offline data fallback.
- advisor_state.json and reports/ are created at runtime when the advisor runs.

## Setup

Python 3.10 or newer is recommended. Create an environment and install the runtime dependencies:

~~~bash
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows PowerShell
.venv\Scripts\Activate.ps1
pip install numpy pandas yfinance
~~~

## Usage

~~~bash
# Run one cycle and exit
python advisor.py --once

# Reproduce the walk-forward report
python advisor.py --validate

# Run as a weekday background service at the configured market-close time
python advisor.py
~~~

The live cycle uses the most recent data available from yfinance. The validation command requests the full history needed for the 2013-2021 training window and the 2022-2026 test window. If the network or yfinance is unavailable, the script falls back to META.csv.

## Research result

The current docstring records an out-of-sample SMA20/SMA150 result for META of Sharpe 1.08, total return 206.6%, and maximum drawdown -22.7%, compared with buy-and-hold Sharpe 0.51, return 78.1%, and maximum drawdown -73.7% over the stated test period. Run advisor.py --validate to recompute the numbers rather than treating them as a guarantee.

## Outputs

- Console recommendation and diagnostic logging.
- advisor_state.json with the simulated position state.
- reports/advisory_META_<date>.md and a CSV activity log.

## Limitations

This is research and paper-trading code for one ticker and one historical window. It does not connect to a broker or place live orders. Historical performance can disappear in a different market regime. Re-run validation on fresh data and review the source before using any output for a financial decision.
