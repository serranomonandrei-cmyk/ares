import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

from ..backtest.engine import BacktestResult
from ..backtest.metrics import equity_curve_data, regime_distribution, monthly_returns


def plot_equity_curve(result: BacktestResult) -> go.Figure:
    df = equity_curve_data(result)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.7, 0.3],
                        vertical_spacing=0.05,
                        subplot_titles=('Equity Curve', 'Drawdown'))

    fig.add_trace(go.Scatter(
        x=df['timestamp'], y=df['equity'],
        mode='lines', name='Equity',
        line=dict(color='#00d4aa', width=2),
        fill='tozeroy', fillcolor='rgba(0,212,170,0.1)',
    ), row=1, col=1)

    peak = df['equity'].cummax()
    drawdown = (df['equity'] - peak) / peak * 100

    fig.add_trace(go.Scatter(
        x=df['timestamp'], y=drawdown,
        mode='lines', name='Drawdown',
        line=dict(color='#ff4444', width=1.5),
        fill='tozeroy', fillcolor='rgba(255,68,68,0.1)',
    ), row=2, col=1)

    fig.add_hline(y=result.initial_capital, line_dash='dash',
                  line_color='gray', row=1, col=1)

    latest_eq = result.final_equity
    init_cap = result.initial_capital
    pnl_color = '#00d4aa' if latest_eq >= init_cap else '#ff4444'

    fig.add_annotation(
        x=df['timestamp'].iloc[-1], y=latest_eq,
        text=f"${latest_eq:.2f}",
        showarrow=True, arrowhead=1,
        font=dict(color=pnl_color, size=12),
        row=1, col=1,
    )

    fig.update_layout(
        height=600,
        template='plotly_dark',
        showlegend=False,
        margin=dict(l=40, r=40, t=40, b=40),
    )
    fig.update_yaxes(title_text='Equity ($)', row=1, col=1)
    fig.update_yaxes(title_text='Drawdown %', row=2, col=1)
    fig.update_xaxes(title_text='Date', row=2, col=1)

    return fig


def plot_trade_pnl(result: BacktestResult) -> go.Figure:
    trades = result.trades
    if not trades:
        return go.Figure()

    df = pd.DataFrame([{
        'entry': t.entry_time,
        'pnl': t.pnl,
        'side': t.side.upper(),
        'pnl_pct': t.pnl_pct * 100,
        'exit_reason': t.exit_reason,
    } for t in trades])

    colors = ['#00d4aa' if p > 0 else '#ff4444' for p in df['pnl']]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df['entry'], y=df['pnl'],
        marker_color=colors,
        hovertemplate='Entry: %{x}<br>PnL: $%{y:.2f}<br>Side: %{customdata[0]}<br>Exit: %{customdata[1]}<extra></extra>',
        customdata=df[['side', 'exit_reason']],
        name='Trade PnL',
    ))

    fig.add_hline(y=0, line_color='gray', line_dash='dash')

    win_count = sum(1 for p in df['pnl'] if p > 0)
    loss_count = sum(1 for p in df['pnl'] if p <= 0)
    total_pnl = df['pnl'].sum()

    fig.add_annotation(
        x=0.02, y=0.95, xref='paper', yref='paper',
        text=f"Wins: {win_count} | Losses: {loss_count} | Net: ${total_pnl:.2f}",
        showarrow=False, font=dict(size=11, color='white'),
        bgcolor='rgba(0,0,0,0.5)', bordercolor='gray', borderwidth=1,
    )

    fig.update_layout(
        height=400, template='plotly_dark',
        title='Trade Profit/Loss',
        xaxis_title='Date', yaxis_title='PnL ($)',
        margin=dict(l=40, r=40, t=40, b=40),
    )
    return fig


def plot_regime_pie(result: BacktestResult) -> go.Figure:
    dist = regime_distribution(result)
    if not dist:
        return go.Figure()

    labels = list(dist.keys())
    values = [v['count'] for v in dist.values()]

    colors = {
        'trending_bullish': '#00d4aa', 'trending_bearish': '#ff4444',
        'strong_trend_bullish': '#00aa88', 'strong_trend_bearish': '#cc3333',
        'ranging_low_vol': '#ffaa00', 'ranging_high_vol': '#ff6600',
        'default': '#888888',
    }

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.4,
        marker=dict(colors=[colors.get(l, '#888888') for l in labels]),
        textinfo='label+percent',
        textposition='outside',
    )])

    fig.update_layout(
        height=400, template='plotly_dark',
        title='Market Regime Distribution',
        margin=dict(l=40, r=40, t=40, b=40),
    )
    return fig


def plot_monthly_returns(result: BacktestResult) -> go.Figure:
    mret = monthly_returns(result)
    if mret.empty:
        return go.Figure()

    colors = ['#00d4aa' if v >= 0 else '#ff4444' for v in mret['Return%']]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=mret.index.astype(str),
        y=mret['Return%'],
        marker_color=colors,
        hovertemplate='%{x}<br>Return: %{y:.1f}%<extra></extra>',
        name='Monthly Return',
    ))

    fig.add_hline(y=0, line_color='gray', line_dash='dash')

    fig.update_layout(
        height=350, template='plotly_dark',
        title='Monthly Returns (%)',
        xaxis_title='Month', yaxis_title='Return %',
        margin=dict(l=40, r=40, t=40, b=40),
    )
    return fig


def plot_regime_performance(result: BacktestResult) -> go.Figure:
    trades = result.trades
    regime_log = result.regime_log
    if not trades or not regime_log:
        return go.Figure()

    regime_at_trade = []
    for t in trades:
        entry_ts = t.entry_time
        best_idx = 0
        for i, ts in enumerate(result.timestamps):
            if ts <= entry_ts:
                best_idx = min(i, len(regime_log)-1)
        r = regime_log[best_idx] if best_idx < len(regime_log) else {}
        regime_at_trade.append(r.get('composite', 'unknown'))

    df = pd.DataFrame({
        'regime': regime_at_trade,
        'pnl': [t.pnl for t in trades],
    })

    perf = df.groupby('regime').agg(
        trades=('pnl', 'count'),
        total_pnl=('pnl', 'sum'),
        avg_pnl=('pnl', 'mean'),
        win_rate=('pnl', lambda x: (x > 0).mean() * 100),
    ).reset_index()

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.5, 0.5],
                        vertical_spacing=0.08)

    colors_reg = ['#00d4aa' if p > 0 else '#ff4444' for p in perf['total_pnl']]

    fig.add_trace(go.Bar(
        x=perf['regime'], y=perf['total_pnl'],
        marker_color=colors_reg,
        name='Total PnL',
        hovertemplate='%{x}<br>Total PnL: $%{y:.2f}<br>Trades: %{customdata}<extra></extra>',
        customdata=perf['trades'],
    ), row=1, col=1)

    fig.add_trace(go.Bar(
        x=perf['regime'], y=perf['win_rate'],
        marker_color='#4488ff',
        name='Win Rate %',
        hovertemplate='%{x}<br>Win Rate: %{y:.1f}%<extra></extra>',
    ), row=2, col=1)

    fig.update_layout(
        height=500, template='plotly_dark',
        title='Performance by Market Regime',
        margin=dict(l=40, r=40, t=40, b=40),
        showlegend=False,
    )
    fig.update_yaxes(title_text='Total PnL ($)', row=1, col=1)
    fig.update_yaxes(title_text='Win Rate %', row=2, col=1)

    return fig
