import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timezone

from ..config import (
    TRADING_PAIRS, INITIAL_CAPITAL, IN_SAMPLE_START, IN_SAMPLE_END,
    OUT_OF_SAMPLE_START, LEVERAGE_DEFAULT, RISK_PER_TRADE, MAX_POSITIONS,
)
from ..data.downloader import download_all_data, list_cached_symbols, load_cached_data
from ..backtest.engine import run_backtest
from ..backtest.metrics import summary_table, trade_list_data, equity_curve_data
from .plots import (
    plot_equity_curve, plot_trade_pnl, plot_regime_pie,
    plot_monthly_returns, plot_regime_performance,
)


def run_dashboard():
    st.set_page_config(
        page_title='ARES - Adaptive Regime Ensemble Strategy',
        page_icon='📊',
        layout='wide',
        initial_sidebar_state='expanded',
    )

    st.markdown("""
        <style>
        .stApp { background-color: #0e1117; }
        .css-1aumxhk, .css-1r6slb0, .css-1wrcr25 { color: #00d4aa; }
        .metric-card {
            background: #1a1d28; border-radius: 10px; padding: 15px;
            border-left: 4px solid #00d4aa; margin-bottom: 10px;
        }
        .metric-label { color: #888; font-size: 0.8rem; }
        .metric-value { color: #fff; font-size: 1.5rem; font-weight: bold; }
        .metric-positive { color: #00d4aa; }
        .metric-negative { color: #ff4444; }
        </style>
    """, unsafe_allow_html=True)

    st.title('🚀 ARES - Adaptive Regime Ensemble Strategy')
    st.markdown('**Multi-timeframe futures trading bot | 1h/4h | Regime-Adaptive | Realistic Backtest**')

    with st.sidebar:
        st.header('⚙️ Configuration')

        mode = st.radio('Mode', ['Backtest', 'Download Data'], index=0)

        symbol = st.selectbox(
            'Trading Pair',
            options=TRADING_PAIRS + ['Custom...'],
            index=0,
        )
        if symbol == 'Custom...':
            symbol = st.text_input('Enter symbol (e.g., BTC/USDT)', 'BTC/USDT')

        timeframe = st.radio('Timeframe', ['1h', '4h', '1h+4h (MTF)'], index=2)

        col1, col2 = st.columns(2)
        with col1:
            capital = st.number_input('Initial Capital ($)', min_value=10.0, value=float(INITIAL_CAPITAL), step=10.0)
        with col2:
            leverage = st.slider('Max Leverage', 1, 15, LEVERAGE_DEFAULT)

        risk_per_trade = st.slider('Risk per Trade (%)', 0.5, 3.0, RISK_PER_TRADE * 100) / 100
        max_positions = st.slider('Max Positions', 1, 10, MAX_POSITIONS)

        st.markdown('---')
        st.markdown('**Data Periods**')
        st.markdown(f'📚 **In-Sample:** {IN_SAMPLE_START} to {IN_SAMPLE_END}')
        st.markdown(f'🧪 **Out-of-Sample:** {OUT_OF_SAMPLE_START} to Present')

        test_period = st.radio('Test on:', ['Out-of-Sample (2026)', 'In-Sample (2023-2025)', 'Full Period'], index=0)

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        '📈 Results', '📊 Trades', '📉 Equity', '🔬 Regime Analysis', '📋 Log'
    ])

    results_container = st.empty()

    if mode == 'Download Data':
        st.info('Downloading data from Binance...')
        with st.spinner('Fetching OHLCV data...'):
            data = download_all_data(symbols=[symbol])
        st.success(f'Downloaded data for {symbol}')
        cached = list_cached_symbols()
        st.write(f'Cached symbols: {cached}')

    else:
        df_1h = load_cached_data(symbol, '1h')
        df_4h = load_cached_data(symbol, '4h')

        if df_1h.empty:
            st.warning(f'No cached data for {symbol}. Go to "Download Data" mode first.')
            st.info('Switching to download mode automatically...')
            with st.spinner('Downloading data...'):
                data = download_all_data(symbols=[symbol])
                if f"{symbol}_1h" in data:
                    df_1h = data[f"{symbol}_1h"]
                    df_4h = data[f"{symbol}_4h"]

        if not df_1h.empty and not df_4h.empty:
            periods = {
                'Out-of-Sample (2026)': {'start': OUT_OF_SAMPLE_START, 'end': None, 'name': 'Out-of-Sample'},
                'In-Sample (2023-2025)': {'start': IN_SAMPLE_START, 'end': IN_SAMPLE_END, 'name': 'In-Sample'},
                'Full Period': {'start': IN_SAMPLE_START, 'end': None, 'name': 'Full'},
            }
            selected = periods[test_period]

            with st.spinner(f'Running backtest on {selected["name"]} data...'):
                result = run_backtest(
                    df_1h=df_1h, df_4h=df_4h,
                    symbol=symbol,
                    initial_capital=capital,
                    leverage=leverage,
                    risk_per_trade=risk_per_trade,
                    max_positions=max_positions,
                    start_date=selected['start'],
                    end_date=selected['end'],
                )

            with tab1:
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    pnl_color = 'metric-positive' if result.total_pnl >= 0 else 'metric-negative'
                    st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-label">Total Return</div>
                            <div class="metric-value {pnl_color}">{result.total_pnl_pct:.2f}%</div>
                            <div style="color:#888;font-size:0.9rem">${result.total_pnl:.2f}</div>
                        </div>
                    """, unsafe_allow_html=True)

                with col2:
                    st.markdown(f"""
                        <div class="metric-card" style="border-left-color:#4488ff">
                            <div class="metric-label">Trades / Win Rate</div>
                            <div class="metric-value">{result.total_trades}</div>
                            <div style="color:#888;font-size:0.9rem">{result.win_rate:.1f}% Win Rate</div>
                        </div>
                    """, unsafe_allow_html=True)

                with col3:
                    dd_color = 'metric-negative' if result.max_drawdown_pct > 15 else 'metric-value'
                    st.markdown(f"""
                        <div class="metric-card" style="border-left-color:#ff4444">
                            <div class="metric-label">Max Drawdown</div>
                            <div class="metric-value" style="color:#ff4444">{result.max_drawdown_pct:.2f}%</div>
                            <div style="color:#888;font-size:0.9rem">${result.max_drawdown:.2f}</div>
                        </div>
                    """, unsafe_allow_html=True)

                with col4:
                    st.markdown(f"""
                        <div class="metric-card" style="border-left-color:#ffaa00">
                            <div class="metric-label">Sharpe / Sortino</div>
                            <div class="metric-value">{result.sharpe_ratio:.2f}</div>
                            <div style="color:#888;font-size:0.9rem">Sortino: {result.sortino_ratio:.2f}</div>
                        </div>
                    """, unsafe_allow_html=True)

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.markdown(f"""
                        <div class="metric-card" style="border-left-color:#aa66ff">
                            <div class="metric-label">Profit Factor</div>
                            <div class="metric-value">{result.profit_factor:.2f}</div>
                        </div>
                    """, unsafe_allow_html=True)
                with col2:
                    st.markdown(f"""
                        <div class="metric-card" style="border-left-color:#66ccff">
                            <div class="metric-label">Calmar Ratio</div>
                            <div class="metric-value">{result.calmar_ratio:.2f}</div>
                        </div>
                    """, unsafe_allow_html=True)
                with col3:
                    st.markdown(f"""
                        <div class="metric-card" style="border-left-color:#88dd88">
                            <div class="metric-label">Avg Win / Avg Loss</div>
                            <div class="metric-value">${result.avg_win:.2f}</div>
                            <div style="color:#888;font-size:0.9rem">Loss: ${result.avg_loss:.2f}</div>
                        </div>
                    """, unsafe_allow_html=True)
                with col4:
                    st.markdown(f"""
                        <div class="metric-card" style="border-left-color:#ff8866">
                            <div class="metric-label">Fees Paid</div>
                            <div class="metric-value">${result.total_fees:.2f}</div>
                            <div style="color:#888;font-size:0.9rem">Avg Hold: {result.avg_hold_bars:.1f}h</div>
                        </div>
                    """, unsafe_allow_html=True)

                st.plotly_chart(plot_equity_curve(result), use_container_width=True)

                col_a, col_b = st.columns(2)
                with col_a:
                    st.plotly_chart(plot_monthly_returns(result), use_container_width=True)
                with col_b:
                    st.plotly_chart(plot_trade_pnl(result), use_container_width=True)

            with tab2:
                trades_df = trade_list_data(result)
                st.dataframe(trades_df, use_container_width=True, hide_index=True)

                if result.total_trades > 0:
                    equity_df = equity_curve_data(result)
                    st.download_button(
                        '📥 Download Results CSV',
                        data=pd.concat([
                            trades_df,
                            equity_df.rename(columns={'equity': 'Equity'})
                        ], axis=1).to_csv(index=False),
                        file_name=f'ares_{symbol}_{selected["name"]}_results.csv',
                        mime='text/csv',
                    )

            with tab3:
                st.plotly_chart(plot_equity_curve(result), use_container_width=True)

                eq_df = equity_curve_data(result)
                st.dataframe(eq_df, use_container_width=True, hide_index=True)

            with tab4:
                col_a, col_b = st.columns(2)
                with col_a:
                    st.plotly_chart(plot_regime_pie(result), use_container_width=True)
                with col_b:
                    st.plotly_chart(plot_regime_performance(result), use_container_width=True)

                dist = result.regime_log
                if dist:
                    st.subheader('Regime Timeline')
                    regime_series = pd.Series([r.get('composite', 'unknown') for r in dist])
                    regime_df = pd.DataFrame({
                        'timestamp': result.timestamps[:len(regime_series)],
                        'regime': regime_series,
                    })
                    st.dataframe(regime_df, use_container_width=True, hide_index=True)

            with tab5:
                st.subheader('Backtest Configuration')
                config_data = {
                    'Parameter': ['Symbol', 'Initial Capital', 'Max Leverage', 'Risk per Trade',
                                  'Max Positions', 'Test Period', 'Data Range', 'Taker Fee',
                                  'Slippage'],
                    'Value': [symbol, f'${capital:.2f}', f'{leverage}x', f'{risk_per_trade*100:.1f}%',
                              str(max_positions), selected['name'],
                              f"{selected['start']} to {selected['end'] or 'Present'}", '0.04%', '0.05%'],
                }
                st.table(pd.DataFrame(config_data))

                if result.total_trades > 0:
                    st.subheader('Trade Summary Statistics')
                    pnls = [t.pnl for t in result.trades]
                    win_pnls = [t.pnl for t in result.trades if t.pnl > 0]
                    loss_pnls = [t.pnl for t in result.trades if t.pnl <= 0]

                    stats = {
                        'Statistic': ['Total Trades', 'Winners', 'Losers', 'Win Rate',
                                      'Largest Win', 'Largest Loss', 'Avg Win', 'Avg Loss',
                                      'Profit Factor', 'Net PnL', 'Return %',
                                      'Max Drawdown %', 'Sharpe Ratio', 'Sortino Ratio'],
                        'Value': [
                            result.total_trades, result.winning_trades, result.losing_trades,
                            f"{result.win_rate:.2f}%",
                            f"${max(win_pnls):.2f}" if win_pnls else 'N/A',
                            f"${min(loss_pnls):.2f}" if loss_pnls else 'N/A',
                            f"${result.avg_win:.2f}", f"${result.avg_loss:.2f}",
                            f"{result.profit_factor:.3f}",
                            f"${result.total_pnl:.2f}", f"{result.total_pnl_pct:.2f}%",
                            f"{result.max_drawdown_pct:.2f}%",
                            f"{result.sharpe_ratio:.3f}", f"{result.sortino_ratio:.3f}",
                        ],
                    }
                    st.table(pd.DataFrame(stats))

        else:
            st.error(f'No data available for {symbol}. Please download data first.')


if __name__ == '__main__':
    run_dashboard()
