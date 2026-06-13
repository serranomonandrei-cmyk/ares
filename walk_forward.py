#!/usr/bin/env python3
"""Walk-forward OOS consistency check: split OOS into 3 windows."""
import sys, os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from crypto_ares.config import (
    IN_SAMPLE_START, IN_SAMPLE_END, OUT_OF_SAMPLE_START,
    INITIAL_CAPITAL, LEVERAGE_DEFAULT, RISK_PER_TRADE,
    ENGULF_BODY, SWING_DIST, INSIDE_VOL, SIGNAL_MIN,
)
from crypto_ares.data.features import compute_all_features
from crypto_ares.data.downloader import resample_4h
from crypto_ares.backtest.engine import run_backtest

SYMBOLS = ['BTC/USDT', 'SOL/USDT', 'BNB/USDT', 'XRP/USDT',
           'ADA/USDT', 'DOGE/USDT', 'AVAX/USDT', 'LINK/USDT', 'DOT/USDT']

SWEET_SPOT_TAG = f"e{ENGULF_BODY}_s{SWING_DIST}_v{INSIDE_VOL}"
ENTRY_TF = '15m'

WINDOWS = [
    ('2025-01-01', '2025-06-01', 'Window 1 (H1 2025)'),
    ('2025-06-01', '2025-11-01', 'Window 2 (H2 2025)'),
    ('2025-11-01', '2026-06-13', 'Window 3 (Nov 25-Jun 26)'),
]


def load_signals(sym_safe, tf):
    path = f"signal_cache/{tf}_{sym_safe}_{SWEET_SPOT_TAG}.npy"
    if not os.path.exists(path):
        return None
    data = np.load(path)
    if data.ndim == 2 and data.shape[1] >= 3:
        return data[:, 0], data[:, 1], data[:, 2]
    return None


def main():
    print("=" * 80)
    print("WALK-FORWARD OOS CONSISTENCY CHECK (15m, no ETH, 2.5% risk)")
    print("=" * 80)

    data = {}
    entry_tfs = [ENTRY_TF, '1h']
    for sym in SYMBOLS:
        safe = sym.replace('/', '_')
        for tf in entry_tfs:
            path = f"data_cache/{safe}_{tf}.parquet"
            if os.path.exists(path):
                df = pd.read_parquet(path)
                if tf not in data:
                    data[tf] = {}
                data[tf][sym] = compute_all_features(df)

    data['4h'] = {}
    for sym in SYMBOLS:
        df_1h = data.get('1h', {}).get(sym)
        if df_1h is not None:
            data['4h'][sym] = compute_all_features(resample_4h(df_1h))

    cached_signals = {}
    for sym in SYMBOLS:
        safe = sym.replace('/', '_')
        sigs = load_signals(safe, ENTRY_TF)
        if sigs is not None:
            cached_signals[sym] = sigs

    all_window_results = {}

    for start, end, label in WINDOWS:
        print(f"\n{'='*60}")
        print(f"  {label}: {start} to {end}")
        print(f"{'='*60}")

        window_results = []
        for sym in SYMBOLS:
            df_entry = data.get(ENTRY_TF, {}).get(sym)
            df_4h = data.get('4h', {}).get(sym)
            df_1h = data.get('1h', {}).get(sym)

            if df_entry is None or df_4h is None:
                continue

            sigs = cached_signals.get(sym)
            cached_tuple = (sigs[0], sigs[1], sigs[2]) if sigs is not None else None

            result = run_backtest(
                df_entry=df_entry, df_4h=df_4h, df_1h=df_1h,
                symbol=sym, initial_capital=INITIAL_CAPITAL,
                leverage=LEVERAGE_DEFAULT, risk_per_trade=RISK_PER_TRADE,
                max_positions=5, start_date=start, end_date=end,
                entry_tf=ENTRY_TF, signal_min=SIGNAL_MIN,
                engulf_body=ENGULF_BODY, swing_dist=SWING_DIST,
                inside_vol=INSIDE_VOL, signal_gate=5, sl_mult=2.0,
                cached_signals=cached_tuple,
            )
            window_results.append(result)

        if not window_results:
            continue

        returns = [r.total_pnl_pct for r in window_results]
        dds = [r.max_drawdown_pct for r in window_results]
        sharpes = [r.sharpe_ratio for r in window_results]
        trades = [r.total_trades for r in window_results]

        avg_ret = np.mean(returns)
        avg_dd = np.mean(dds)
        avg_sharpe = np.mean(sharpes)
        total_trades = sum(trades)
        profitable = sum(1 for r in returns if r > 0)

        all_window_results[label] = {
            'avg_return': avg_ret,
            'avg_dd': avg_dd,
            'avg_sharpe': avg_sharpe,
            'total_trades': total_trades,
            'profitable': profitable,
            'total_coins': len(window_results),
        }

        print(f"  Avg Return: {avg_ret:+.1f}% | Avg DD: {avg_dd:.1f}% | Avg Sharpe: {avg_sharpe:.2f}")
        print(f"  Trades: {total_trades} | Profitable: {profitable}/{len(window_results)}")

        for r in window_results:
            print(f"    {r.symbol}: {r.total_pnl_pct:+.1f}% | DD: {r.max_drawdown_pct:.1f}% | Sharpe: {r.sharpe_ratio:.2f}")

    print(f"\n{'='*80}")
    print("WALK-FORWARD SUMMARY")
    print(f"{'='*80}")

    for label, stats in all_window_results.items():
        print(f"  {label}: Return={stats['avg_return']:+.1f}% | DD={stats['avg_dd']:.1f}% | Sharpe={stats['avg_sharpe']:.2f}")

    all_returns = [s['avg_return'] for s in all_window_results.values()]
    all_sharpes = [s['avg_sharpe'] for s in all_window_results.values()]
    print(f"\n  Consistency: {'PASS' if all(r > 0 for r in all_returns) else 'FAIL'}")
    print(f"  Return range: {min(all_returns):+.1f}% to {max(all_returns):+.1f}%")
    print(f"  Sharpe range: {min(all_sharpes):.2f} to {max(all_sharpes):.2f}")


if __name__ == '__main__':
    main()
