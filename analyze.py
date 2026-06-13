#!/usr/bin/env python3
"""
Final analysis: run engine with sweet-spot config (2.5% risk, 30m, e0.7_s6_v1.3),
produce per-coin daily/monthly breakdowns + full summary.
"""
import sys, os, json, math
from datetime import datetime
from pathlib import Path

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
from crypto_ares.backtest.metrics import summary_table, equity_curve_data, trade_list_data, monthly_returns

SYMBOLS = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT', 'XRP/USDT',
           'ADA/USDT', 'DOGE/USDT', 'AVAX/USDT', 'LINK/USDT', 'DOT/USDT']

SWEET_SPOT_TAG = f"e{ENGULF_BODY}_s{SWING_DIST}_v{INSIDE_VOL}"

# Entry timeframe: pass '30m' or '15m' as first arg
ENTRY_TF = sys.argv[1] if len(sys.argv) > 1 else '30m'
# Exclude symbols: pass comma-separated list as second arg
EXCLUDE = set(sys.argv[2].split(',')) if len(sys.argv) > 2 and sys.argv[2] else set()
# Risk per trade: pass as third arg (default from config)
RISK_OVERRIDE = float(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3] else None
# ADX minimum: pass as fourth arg (0 = no filter)
ADX_MIN = float(sys.argv[4]) if len(sys.argv) > 4 and sys.argv[4] else 0.0
RESULT_SUFFIX = f"oos_{ENTRY_TF}"
RESULTS_DIR = Path('results')
RESULTS_DIR.mkdir(exist_ok=True)

def load_sweet_spot_signals(sym_safe: str, tf: str):
    """Load cached GPU signals for sweet-spot params."""
    path = f"signal_cache/{tf}_{sym_safe}_{SWEET_SPOT_TAG}.npy"
    if not os.path.exists(path):
        print(f"    Signal cache not found: {path}")
        return None
    data = np.load(path)
    # Expect shape (n, 3) or separate files
    if data.ndim == 2 and data.shape[1] >= 3:
        return data[:, 0], data[:, 1], data[:, 2]
    # Try separate files
    long_path = path.replace('.npy', '_long.npy')
    short_path = path.replace('.npy', '_short.npy')
    conf_path = path.replace('.npy', '_conf.npy')
    if os.path.exists(long_path):
        return np.load(long_path), np.load(short_path), np.load(conf_path)
    return None

def compute_portfolio_stats(oos_results, entry_tf='30m'):
    """Compute portfolio-level metrics from individual results."""
    total_eq = sum(r.final_equity for r in oos_results)
    total_init = sum(r.initial_capital for r in oos_results)
    port_return_pct = (total_eq - total_init) / total_init * 100

    # Portfolio equity curve (sum of individual)
    min_len = min(len(r.equity_curve) for r in oos_results if r.trades)
    if min_len == 0:
        return {'return_pct': port_return_pct}

    # Align by timestamps - use the union of all timestamps
    all_ts = sorted(set(ts for r in oos_results for ts in r.timestamps))
    eq_map = {}
    for r in oos_results:
        for ts, eq in zip(r.timestamps, r.equity_curve):
            eq_map[ts] = eq_map.get(ts, 0) + eq

    curve = pd.Series([eq_map[ts] for ts in all_ts], index=all_ts)
    curve_pct = curve.pct_change().dropna()

    peak = curve.cummax()
    dd = (curve - peak) / peak
    max_dd = float(dd.min() * 100)

    bars_per_day = {'15m': 96, '30m': 48, '1h': 24}.get(entry_tf, 96)
    ann_factor = bars_per_day * 365
    sharpe = float(curve_pct.mean() / curve_pct.std() * math.sqrt(ann_factor)) if curve_pct.std() > 0 else 0
    down = curve_pct[curve_pct < 0]
    sortino = float(curve_pct.mean() / down.std() * math.sqrt(ann_factor)) if len(down) > 0 and down.std() > 0 else 0
    calmar = port_return_pct / abs(max_dd) if max_dd != 0 else 0

    return {
        'return_pct': port_return_pct,
        'max_dd_pct': max_dd,
        'sharpe': sharpe,
        'sortino': sortino,
        'calmar': calmar,
    }

def compute_daily_breakdown(result):
    """Per-coin daily win rate and PnL."""
    if not result.trades:
        return pd.DataFrame()
    rows = []
    for t in result.trades:
        day = pd.Timestamp(t.exit_time).strftime('%Y-%m-%d') if t.exit_time else 'unknown'
        rows.append({'date': day, 'pnl': t.pnl, 'pnl_pct': t.pnl_pct})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df['date'] = pd.to_datetime(df['date'])
    daily = df.groupby('date').agg(
        trades=('pnl', 'count'),
        total_pnl=('pnl', 'sum'),
        winning=('pnl', lambda x: (x > 0).sum()),
    )
    daily['win_rate'] = daily['winning'] / daily['trades'] * 100
    daily['cum_pnl'] = daily['total_pnl'].cumsum()
    return daily

def compute_trade_stats(result):
    """Detailed trade-level stats."""
    trades = result.trades
    if not trades:
        return {}
    df = pd.DataFrame([{
        'side': t.side,
        'pnl': t.pnl,
        'pnl_pct': t.pnl_pct,
        'entry_sig': t.entry_signal_strength,
        'exit_reason': t.exit_reason,
        'hold_h': (pd.Timestamp(t.exit_time) - pd.Timestamp(t.entry_time)).total_seconds() / 3600 if t.exit_time else 0,
    } for t in trades])

    return {
        'total_trades': len(df),
        'long_trades': int((df['side'] == 'long').sum()),
        'short_trades': int((df['side'] == 'short').sum()),
        'long_win_rate': float(df[df['side'] == 'long']['pnl'].gt(0).mean() * 100) if (df['side'] == 'long').sum() > 0 else 0,
        'short_win_rate': float(df[df['side'] == 'short']['pnl'].gt(0).mean() * 100) if (df['side'] == 'short').sum() > 0 else 0,
        'avg_win': float(df[df['pnl'] > 0]['pnl'].mean()) if (df['pnl'] > 0).sum() > 0 else 0,
        'avg_loss': float(abs(df[df['pnl'] <= 0]['pnl'].mean())) if (df['pnl'] <= 0).sum() > 0 else 0,
        'max_win': float(df['pnl'].max()),
        'max_loss': float(df['pnl'].min()),
        'avg_hold_hours': float(df['hold_h'].mean()),
        'win_by_sl': int(((df['exit_reason'] == 'sl') & (df['pnl'] > 0)).sum()),
        'loss_by_tp': int(((df['exit_reason'] == 'tp') & (df['pnl'] <= 0)).sum()),
        'exit_reasons': df['exit_reason'].value_counts().to_dict(),
        'avg_entry_signal': float(df['entry_sig'].mean()),
    }

def main():
    print("=" * 80)
    print(f"ARES - FINAL ANALYSIS ({ENTRY_TF}, SWEET-SPOT CONFIG)")
    print(f"  Risk: {RISK_PER_TRADE*100:.1f}% | Params: {SWEET_SPOT_TAG} | SigMin: {SIGNAL_MIN}")
    print(f"  IS: {IN_SAMPLE_START} to {IN_SAMPLE_END}")
    print(f"  OOS: {OUT_OF_SAMPLE_START} to Present")
    print("=" * 80)

    print(f"\n[1/3] Loading {ENTRY_TF} data for {len(SYMBOLS)} symbols...")
    data = {}
    entry_tfs = [ENTRY_TF, '1h']
    for sym in SYMBOLS:
        if sym in EXCLUDE:
            continue
        safe = sym.replace('/', '_').replace(' ', '_')
        for tf in entry_tfs:
            path = f"data_cache/{safe}_{tf}.parquet"
            if os.path.exists(path):
                df = pd.read_parquet(path)
                if tf not in data:
                    data[tf] = {}
                data[tf][sym] = compute_all_features(df)

    # Resample 4h from 1h (4h is not cached as separate parquet)
    data['4h'] = {}
    for sym in SYMBOLS:
        if sym in EXCLUDE:
            continue
        df_1h = data.get('1h', {}).get(sym)
        if df_1h is not None:
            df_4h = resample_4h(df_1h)
            data['4h'][sym] = compute_all_features(df_4h)

    n_entry = len(data.get(ENTRY_TF, {}))
    print(f"  {ENTRY_TF}: {n_entry} symbols")
    print(f"  4h:  {len(data.get('4h', {}))} symbols")
    print(f"  1h:  {len(data.get('1h', {}))} symbols")

    print(f"\n[2/3] Loading sweet-spot cached signals ({SWEET_SPOT_TAG}, {ENTRY_TF})...")
    cached_signals = {}
    for sym in SYMBOLS:
        safe = sym.replace('/', '_').replace(' ', '_')
        sigs = load_sweet_spot_signals(safe, ENTRY_TF)
        if sigs is not None:
            cached_signals[sym] = sigs
            print(f"  {sym}: signals loaded ({len(sigs[0])} bars)")
        else:
            print(f"  {sym}: NO CACHED SIGNALS - will compute on the fly")

    print(f"\n[3/3] Running OOS backtest for each coin ({ENTRY_TF})...")
    effective_risk = RISK_OVERRIDE if RISK_OVERRIDE else RISK_PER_TRADE
    print(f"  Risk: {effective_risk*100:.1f}% | Leverage: {LEVERAGE_DEFAULT}x")
    print(f"  Entry TF: {ENTRY_TF} | Signal gate: 5/day (scales with equity)")
    print()

    oos_results = []
    all_daily_breakdowns = {}

    for sym in SYMBOLS:
        if sym in EXCLUDE:
            continue
        df_entry = data.get(ENTRY_TF, {}).get(sym)
        df_4h = data.get('4h', {}).get(sym)
        df_1h = data.get('1h', {}).get(sym)

        if df_entry is None or df_4h is None:
            print(f"  {sym}: SKIP (missing data)")
            continue

        sigs = cached_signals.get(sym)
        cached_tuple = (sigs[0], sigs[1], sigs[2]) if sigs is not None else None

        print(f"  [{sym}] Running...", end=' ', flush=True)
        result = run_backtest(
            df_entry=df_entry,
            df_4h=df_4h,
            df_1h=df_1h,
            symbol=sym,
            initial_capital=INITIAL_CAPITAL,
            leverage=LEVERAGE_DEFAULT,
            risk_per_trade=effective_risk,
            max_positions=5,
            start_date=OUT_OF_SAMPLE_START,
            end_date=None,
            entry_tf=ENTRY_TF,
            signal_min=SIGNAL_MIN,
            engulf_body=ENGULF_BODY,
            swing_dist=SWING_DIST,
            inside_vol=INSIDE_VOL,
            signal_gate=5,
            sl_mult=2.0,
            cached_signals=cached_tuple,
            adx_min=ADX_MIN,
        )
        oos_results.append(result)

        daily = compute_daily_breakdown(result)
        all_daily_breakdowns[sym] = daily

        ts = result
        print(f"Ret: {ts.total_pnl_pct:+.1f}% | Trades: {ts.total_trades} | "
              f"Win: {ts.win_rate:.1f}% | PF: {ts.profit_factor:.2f} | "
              f"DD: {ts.max_drawdown_pct:.1f}% | Sharpe: {ts.sharpe_ratio:.2f}")

    # ============================================================
    # SAVE RESULTS
    # ============================================================
    print(f"\n{'='*80}")
    print("SAVING RESULTS")
    print('='*80)

    # 1. Summary table
    summary_df = summary_table(oos_results)
    summary_path = RESULTS_DIR / f'{RESULT_SUFFIX}_summary.csv'
    summary_df.to_csv(summary_path, index=False)
    print(f"  Summary saved: {summary_path}")

    # 2. Per-coin breakdowns
    for sym in SYMBOLS:
        daily = all_daily_breakdowns.get(sym)
        if daily is not None and not daily.empty:
            safe = sym.replace('/', '_')
            daily.to_csv(RESULTS_DIR / f'daily_{safe}_{ENTRY_TF}.csv')
            print(f"  Daily breakdown saved: {safe} ({len(daily)} trading days)")

    # 3. Trade lists
    for r in oos_results:
        safe = r.symbol.replace('/', '_')
        trades_df = trade_list_data(r)
        if not trades_df.empty:
            trades_df.to_csv(RESULTS_DIR / f'trades_{safe}_{ENTRY_TF}.csv', index=False)

    # 4. Equity curves
    for r in oos_results:
        safe = r.symbol.replace('/', '_')
        eq_df = equity_curve_data(r)
        eq_df.to_csv(RESULTS_DIR / f'equity_{safe}_{ENTRY_TF}.csv', index=False)

    # 5. Monthly returns
    monthly_all = {}
    for r in oos_results:
        safe = r.symbol.replace('/', '_')
        monthly = monthly_returns(r)
        if not monthly.empty:
            monthly.to_csv(RESULTS_DIR / f'monthly_{safe}_{ENTRY_TF}.csv')
            monthly_all[safe] = monthly

    # 6. Portfolio stats
    port_stats = compute_portfolio_stats(oos_results, ENTRY_TF)
    with open(RESULTS_DIR / f'portfolio_stats_{ENTRY_TF}.json', 'w') as f:
        json.dump(port_stats, f, indent=2)
    print(f"  Portfolio stats saved")

    # 7. Trade stats per coin
    for r in oos_results:
        safe = r.symbol.replace('/', '_')
        stats = compute_trade_stats(r)
        with open(RESULTS_DIR / f'trade_stats_{safe}_{ENTRY_TF}.json', 'w') as f:
            json.dump(stats, f, indent=2)

    # ============================================================
    # PRINT FINAL REPORT
    # ============================================================
    print(f"\n{'='*80}")
    print(f"OUT-OF-SAMPLE RESULTS SUMMARY ({ENTRY_TF})")
    print(f"  Period: {OUT_OF_SAMPLE_START} to Present")
    print(f"  Config: {ENTRY_TF}, {RISK_PER_TRADE*100:.1f}% risk, {LEVERAGE_DEFAULT}x leverage")
    print(f"  Params: engulf={ENGULF_BODY}, swing={SWING_DIST}, inside_vol={INSIDE_VOL}, sig_min={SIGNAL_MIN}")
    print('='*80)
    print(summary_df.to_string(index=False))

    avg_ret = np.mean([r.total_pnl_pct for r in oos_results])
    avg_win = np.mean([r.win_rate for r in oos_results])
    avg_pf = np.mean([r.profit_factor for r in oos_results])
    avg_dd = np.mean([r.max_drawdown_pct for r in oos_results])
    avg_sharpe = np.mean([r.sharpe_ratio for r in oos_results])
    positive = sum(1 for r in oos_results if r.total_pnl > 0)

    print(f"\n  {'-'*60}")
    print(f"  AVERAGES:")
    print(f"    Return:       {avg_ret:+.2f}%")
    print(f"    Win Rate:     {avg_win:.1f}%")
    print(f"    Profit Factor: {avg_pf:.2f}")
    print(f"    Max DD:       {avg_dd:.1f}%")
    print(f"    Sharpe:       {avg_sharpe:.2f}")
    print(f"    Profitable:   {positive}/{len(oos_results)} coins")
    print(f"  {'-'*60}")

    print(f"\n  PORTFOLIO (equal-weighted):")
    print(f"    Total Return:  {port_stats['return_pct']:+.2f}%")
    print(f"    Max DD:        {port_stats.get('max_dd_pct', 0):.1f}%")
    print(f"    Sharpe:        {port_stats.get('sharpe', 0):.2f}")
    print(f"    Sortino:       {port_stats.get('sortino', 0):.2f}")
    print(f"    Calmar:        {port_stats.get('calmar', 0):.2f}")
    print(f"  {'-'*60}")

    print(f"\n{'='*80}")
    print("PER-COIN DETAILED BREAKDOWN")
    print('='*80)

    for r in oos_results:
        safe = r.symbol.replace('/', '_')
        stats = compute_trade_stats(r)
        daily = all_daily_breakdowns.get(r.symbol)

        print(f"\n  [{r.symbol}]")
        print(f"    Return: {r.total_pnl_pct:+.2f}% | Trades: {r.total_trades} "
              f"| Win: {r.win_rate:.1f}% | PF: {r.profit_factor:.2f} | DD: {r.max_drawdown_pct:.1f}%")
        print(f"    Long: {stats['long_trades']} ({stats['long_win_rate']:.0f}% WR) | "
              f"Short: {stats['short_trades']} ({stats['short_win_rate']:.0f}% WR)")
        print(f"    Avg Win: ${stats['avg_win']:.2f} | Avg Loss: ${stats['avg_loss']:.2f} | "
              f"Max Win: ${stats['max_win']:.2f} | Max Loss: ${stats['max_loss']:.2f}")
        print(f"    Avg Hold: {stats['avg_hold_hours']:.1f}h | Avg Entry Sig: {stats['avg_entry_signal']:.2f}")
        print(f"    Exit Reasons: {stats['exit_reasons']}")
        if daily is not None and not daily.empty:
            best_day = daily.loc[daily['total_pnl'].idxmax()]
            worst_day = daily.loc[daily['total_pnl'].idxmin()]
            print(f"    Best Day: {best_day.name.strftime('%Y-%m-%d')} (${best_day['total_pnl']:.2f}) | "
                  f"Worst Day: {worst_day.name.strftime('%Y-%m-%d')} (${worst_day['total_pnl']:.2f})")
            win_days = (daily['total_pnl'] > 0).sum()
            total_days = len(daily)
            print(f"    Profitable Days: {win_days}/{total_days} ({win_days/total_days*100:.0f}%)")

    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print(f"All results saved to {RESULTS_DIR}/")
    print('='*80)


if __name__ == '__main__':
    main()
