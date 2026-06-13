import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from crypto_ares.backtest.engine import run_backtest
from crypto_ares.data.downloader import load_cached_data

df_1h = load_cached_data('BTC/USDT', '1h')
df_4h = load_cached_data('BTC/USDT', '4h')
print(f'Data: 1h={len(df_1h)}, 4h={len(df_4h)}')

result = run_backtest(df_1h, df_4h, symbol='BTC/USDT', start_date='2026-01-01', end_date=None)
print(f'OOS Result: PnL={result.total_pnl_pct:.2f}% Trades={result.total_trades} Win={result.win_rate:.1f}% PF={result.profit_factor:.2f}')
print(f'Equity curve points: {len(result.equity_curve)}, Trades logged: {len(result.trades)}')

if result.trades:
    t = result.trades[0]
    print(f'First trade: side={t.side} entry_price={t.entry_price:.2f} exit_price={t.exit_price:.2f} pnl={t.pnl:.2f}')

from crypto_ares.app.plots import plot_equity_curve, plot_trade_pnl
fig1 = plot_equity_curve(result)
fig2 = plot_trade_pnl(result)
print(f'Equity curve figure OK: {len(fig1.data) > 0}')
print(f'Trade PnL figure OK: {len(fig2.data) > 0}')

from crypto_ares.backtest.metrics import summary_table, trade_list_data, regime_distribution
metrics_df = summary_table([result])
print(f'Summary table: {len(metrics_df)} rows')
trades_df = trade_list_data(result)
print(f'Trade list: {len(trades_df)} rows')
dist = regime_distribution(result)
print(f'Regime distribution: {dist.keys() if dist else "empty"}')

print('ALL INTEGRATION TESTS PASSED')
