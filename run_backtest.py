#!/usr/bin/env python3
"""
Full pipeline: Download data -> Run in-sample optimization -> Run out-of-sample test -> Print results
"""

import sys
import os
import time
import json
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from crypto_ares.config import (
    TRADING_PAIRS, INITIAL_CAPITAL, LEVERAGE_DEFAULT,
    IN_SAMPLE_START, IN_SAMPLE_END, OUT_OF_SAMPLE_START,
    RISK_PER_TRADE, MAX_POSITIONS,
)
from crypto_ares.data.downloader import download_all_data, load_cached_data
from crypto_ares.backtest.engine import run_backtest
from crypto_ares.backtest.metrics import summary_table

SYMBOLS = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT', 'XRP/USDT',
           'ADA/USDT', 'DOGE/USDT', 'AVAX/USDT', 'LINK/USDT', 'DOT/USDT']

def main():
    print("=" * 80)
    print("ARES - Adaptive Regime Ensemble Strategy")
    print("Full Backtest Pipeline")
    print("=" * 80)

    print(f"\n[1/3] Downloading data for {len(SYMBOLS)} symbols...")
    data = download_all_data(symbols=SYMBOLS)

    print(f"\n[2/3] Running in-sample backtest ({IN_SAMPLE_START} to {IN_SAMPLE_END})...")
    is_results = []
    for sym in SYMBOLS:
        df_1h = data.get(f"{sym}_1h")
        df_4h = data.get(f"{sym}_4h")
        if df_1h is None or df_4h is None:
            print(f"  Skipping {sym} - no data")
            continue
        print(f"  Testing {sym}...", end=' ', flush=True)
        result = run_backtest(
            df_1h, df_4h, symbol=sym,
            initial_capital=INITIAL_CAPITAL,
            leverage=LEVERAGE_DEFAULT,
            risk_per_trade=RISK_PER_TRADE,
            max_positions=MAX_POSITIONS,
            start_date=IN_SAMPLE_START,
            end_date=IN_SAMPLE_END,
        )
        is_results.append(result)
        print(f"Return: {result.total_pnl_pct:+.2f}% | Trades: {result.total_trades} | Win: {result.win_rate:.1f}% | PF: {result.profit_factor:.2f} | DD: {result.max_drawdown_pct:.1f}%")

    print(f"\n[3/3] Running out-of-sample backtest ({OUT_OF_SAMPLE_START} to Present)...")
    oos_results = []
    for sym in SYMBOLS:
        df_1h = data.get(f"{sym}_1h")
        df_4h = data.get(f"{sym}_4h")
        if df_1h is None or df_4h is None:
            continue
        print(f"  Testing {sym}...", end=' ', flush=True)
        result = run_backtest(
            df_1h, df_4h, symbol=sym,
            initial_capital=INITIAL_CAPITAL,
            leverage=LEVERAGE_DEFAULT,
            risk_per_trade=RISK_PER_TRADE,
            max_positions=MAX_POSITIONS,
            start_date=OUT_OF_SAMPLE_START,
            end_date=None,
        )
        oos_results.append(result)
        print(f"Return: {result.total_pnl_pct:+.2f}% | Trades: {result.total_trades} | Win: {result.win_rate:.1f}% | PF: {result.profit_factor:.2f} | DD: {result.max_drawdown_pct:.1f}%")

    print("\n" + "=" * 80)
    print("IN-SAMPLE RESULTS SUMMARY")
    print("=" * 80)
    if is_results:
        is_df = summary_table(is_results)
        print(is_df.to_string(index=False))

        avg_ret = sum(r.total_pnl_pct for r in is_results) / len(is_results)
        avg_win = sum(r.win_rate for r in is_results) / len(is_results)
        avg_pf = sum(r.profit_factor for r in is_results) / len(is_results)
        avg_dd = sum(r.max_drawdown_pct for r in is_results) / len(is_results)
        print(f"\n  Average: Return={avg_ret:.2f}% | Win={avg_win:.1f}% | PF={avg_pf:.2f} | DD={avg_dd:.1f}%")

    print("\n" + "=" * 80)
    print("OUT-OF-SAMPLE RESULTS SUMMARY")
    print("=" * 80)
    if oos_results:
        oos_df = summary_table(oos_results)
        print(oos_df.to_string(index=False))

        avg_ret = sum(r.total_pnl_pct for r in oos_results) / len(oos_results)
        avg_win = sum(r.win_rate for r in oos_results) / len(oos_results)
        avg_pf = sum(r.profit_factor for r in oos_results) / len(oos_results)
        avg_dd = sum(r.max_drawdown_pct for r in oos_results) / len(oos_results)
        print(f"\n  Average: Return={avg_ret:.2f}% | Win={avg_win:.1f}% | PF={avg_pf:.2f} | DD={avg_dd:.1f}%")

        oos_returns = [r.total_pnl_pct for r in oos_results]
        positive_oos = sum(1 for r in oos_results if r.total_pnl > 0)
        print(f"\n  OOS Performance: {positive_oos}/{len(oos_results)} symbols profitable")

    print("\nDone!")


if __name__ == '__main__':
    main()
