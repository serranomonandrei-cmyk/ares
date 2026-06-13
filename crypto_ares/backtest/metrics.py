import numpy as np
import pandas as pd
from typing import List

from .engine import BacktestResult

def summary_table(results: List[BacktestResult]) -> pd.DataFrame:
    rows = []
    for r in results:
        rows.append({
            'Symbol': r.symbol,
            'Trades': r.total_trades,
            'Win%': f"{r.win_rate:.1f}%",
            'Net PnL': f"${r.total_pnl:.2f}",
            'Return%': f"{r.total_pnl_pct:.1f}%",
            'Profit Factor': f"{r.profit_factor:.2f}",
            'Max DD%': f"{r.max_drawdown_pct:.1f}%",
            'Sharpe': f"{r.sharpe_ratio:.2f}",
            'Sortino': f"{r.sortino_ratio:.2f}",
            'Calmar': f"{r.calmar_ratio:.2f}",
            'Avg Win': f"${r.avg_win:.2f}",
            'Avg Loss': f"${r.avg_loss:.2f}",
            'Fees': f"${r.total_fees:.2f}",
            'Avg Hold(h)': f"{r.avg_hold_bars:.1f}",
        })
    return pd.DataFrame(rows)


def equity_curve_data(result: BacktestResult) -> pd.DataFrame:
    return pd.DataFrame({
        'timestamp': result.timestamps,
        'equity': result.equity_curve,
    })


def trade_list_data(result: BacktestResult) -> pd.DataFrame:
    rows = []
    for t in result.trades:
        rows.append({
            'Side': t.side.upper(),
            'Entry': t.entry_time,
            'Exit': t.exit_time,
            'Entry Price': f"${t.entry_price:.2f}",
            'Exit Price': f"${t.exit_price:.2f}",
            'Qty': f"{t.quantity:.4f}",
            'Lev': f"{t.leverage:.1f}x",
            'PnL': f"${t.pnl:.2f}",
            'PnL%': f"{t.pnl_pct*100:.2f}%",
            'Exit Reason': t.exit_reason,
            'Fees': f"${t.fee_paid:.4f}",
        })
    return pd.DataFrame(rows)


def regime_distribution(result: BacktestResult) -> dict:
    regimes = [r.get('composite', 'unknown') for r in result.regime_log if r]
    if not regimes:
        return {}
    from collections import Counter
    dist = Counter(regimes)
    total = len(regimes)
    return {k: {'count': v, 'pct': v/total*100} for k, v in dist.most_common()}


def monthly_returns(result: BacktestResult) -> pd.DataFrame:
    if len(result.timestamps) != len(result.equity_curve):
        return pd.DataFrame()
    df = pd.DataFrame({'timestamp': result.timestamps, 'equity': result.equity_curve})
    df['month'] = df['timestamp'].dt.to_period('M')
    monthly = df.groupby('month').agg(first_equity=('equity', 'first'), last_equity=('equity', 'last'))
    monthly['return'] = (monthly['last_equity'] / monthly['first_equity'] - 1) * 100
    return monthly[['return']].rename(columns={'return': 'Return%'})
