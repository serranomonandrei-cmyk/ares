import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Optional

from ..config import TAKER_FEE, SLIPPAGE, INITIAL_CAPITAL, MAX_DRAWDOWN_SOFT, MAX_DRAWDOWN_HARD, RISK_PER_TRADE, MAX_POSITIONS
from ..data.features import compute_all_features
from ..strategy.ensemble import compute_position_size
from ..strategy.regime import detect_regime

try:
    from ..strategy.setups_gpu import compute_all_signals
    _HAS_GPU_SIGNALS = True
except ImportError:
    from ..strategy.setups import evaluate_all_setups
    _HAS_GPU_SIGNALS = False


@dataclass
class Trade:
    symbol: str = ''
    side: str = ''
    entry_time: any = None
    entry_price: float = 0.0
    quantity: float = 0.0
    position_value: float = 0.0
    leverage: float = 1.0
    entry_signal_strength: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    exit_time: any = None
    exit_price: float = 0.0
    pnl: float = 0.0
    pnl_pct: float = 0.0
    exit_reason: str = ''
    fee_paid: float = 0.0


@dataclass
class BacktestResult:
    symbol: str = ''
    timeframe: str = ''
    initial_capital: float = 0.0
    final_equity: float = 0.0
    total_pnl: float = 0.0
    total_pnl_pct: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    total_fees: float = 0.0
    avg_hold_bars: float = 0.0
    trades: list = field(default_factory=list)
    equity_curve: list = field(default_factory=list)
    timestamps: list = field(default_factory=list)
    regime_log: list = field(default_factory=list)


def run_backtest(
    df_entry: pd.DataFrame,
    df_4h: pd.DataFrame,
    df_1h: pd.DataFrame = None,
    symbol: str = 'BTC/USDT',
    initial_capital: float = INITIAL_CAPITAL,
    leverage: int = 10,
    risk_per_trade: float = RISK_PER_TRADE,
    max_positions: int = MAX_POSITIONS,
    start_date: str = None,
    end_date: str = None,
    entry_tf: str = '1h',
    signal_min: float = 0.35,
    engulf_body: float = 0.6,
    swing_dist: int = 4,
    inside_vol: float = 1.3,
    sl_mult: float = 2.0,
    signal_gate: int = 2,
    use_mtf_confirmation: bool = False,
    cached_signals: tuple = None,
    adx_min: float = 0.0,
) -> BacktestResult:
    df_entry = compute_all_features(df_entry.copy())
    df_4h = compute_all_features(df_4h.copy())
    if df_1h is not None:
        df_1h = compute_all_features(df_1h.copy())

    if start_date:
        df_entry = df_entry[df_entry['timestamp'] >= start_date].copy()
    if end_date:
        df_entry = df_entry[df_entry['timestamp'] <= end_date].copy()

    df_entry = df_entry.dropna().sort_values('timestamp').reset_index(drop=True)
    df_4h = df_4h.dropna().sort_values('timestamp').reset_index(drop=True)
    if df_1h is not None:
        df_1h = df_1h.dropna().sort_values('timestamp').reset_index(drop=True)

    min_bars = 200
    if len(df_entry) < min_bars or len(df_4h) < min_bars:
        return _empty_result(symbol, initial_capital, df_entry, entry_tf)

    equity = float(initial_capital)
    peak_equity = equity
    max_drawdown_val = 0.0
    trades = []
    equity_curve = [equity]
    timestamps = [df_entry['timestamp'].iloc[0]]
    regime_log = []
    open_pos = None
    trading_stopped = False
    consecutive_losses = 0
    total_fees = 0.0
    base_risk_per_trade = risk_per_trade
    daily_signal_counts = {}

    recs_entry = df_entry.to_dict('records')
    recs_4h = df_4h.to_dict('records')
    recs_1h = df_1h.to_dict('records') if df_1h is not None else None

    last_regime_update = -1
    cached_4h_idx = min_bars
    cached_regime = detect_regime(df_4h.iloc[:cached_4h_idx])

    signal_long_arr = np.zeros(len(df_entry), dtype=np.float32)
    signal_short_arr = np.zeros(len(df_entry), dtype=np.float32)
    signal_conf_arr = np.zeros(len(df_entry), dtype=np.float32)
    active_setups_arr = np.zeros(len(df_entry), dtype=np.int32)

    if cached_signals is not None:
        sig_long, sig_short, sig_conf = cached_signals
        signal_long_arr[:] = sig_long[-len(df_entry):] if len(sig_long) >= len(df_entry) else np.pad(sig_long, (len(df_entry)-len(sig_long), 0), 'constant')
        signal_short_arr[:] = sig_short[-len(df_entry):] if len(sig_short) >= len(df_entry) else np.pad(sig_short, (len(df_entry)-len(sig_short), 0), 'constant')
        signal_conf_arr[:] = sig_conf[-len(df_entry):] if len(sig_conf) >= len(df_entry) else np.pad(sig_conf, (len(df_entry)-len(sig_conf), 0), 'constant')
    elif _HAS_GPU_SIGNALS:
        print(f"    Computing GPU-based signals (engulf={engulf_body}, swing={swing_dist}, inside_vol={inside_vol})...", flush=True)
        sig_long, sig_short, sig_conf = compute_all_signals(
            df_entry, engulf_body=engulf_body, swing_dist=swing_dist, inside_vol=inside_vol)
        signal_long_arr[:] = sig_long
        signal_short_arr[:] = sig_short
        signal_conf_arr[:] = sig_conf

    setup_window = 45

    for i in range(min_bars, len(recs_entry)):
        row = recs_entry[i]
        ts = row['timestamp']
        price = float(row['close'])

        if trading_stopped:
            equity_curve.append(equity)
            timestamps.append(ts)
            continue

        # Previous COMPLETED 4h bar — the current 4h bucket is not yet finished
        ts_4h = ts.replace(minute=ts.minute // 240 * 240, second=0, microsecond=0) if hasattr(ts, 'minute') else ts
        ts_4h = pd.Timestamp(ts_4h).floor('4h') if not isinstance(ts_4h, pd.Timestamp) else ts_4h.floor('4h')
        ts_4h -= pd.Timedelta(hours=4)

        curr_4h_idx = None
        for j in range(len(recs_4h)):
            if recs_4h[j]['timestamp'] <= ts_4h:
                curr_4h_idx = j
            else:
                break
        if curr_4h_idx is None or curr_4h_idx < min_bars:
            equity_curve.append(equity)
            timestamps.append(ts)
            continue

        if curr_4h_idx != last_regime_update:
            cached_4h_idx = curr_4h_idx
            cached_regime = detect_regime(df_4h.iloc[:cached_4h_idx + 1])
            last_regime_update = curr_4h_idx

        regime_log.append(cached_regime)

        direction_filter = cached_regime.get('direction', 'neutral')

        if _HAS_GPU_SIGNALS:
            sig_long_val = float(signal_long_arr[i])
            sig_short_val = float(signal_short_arr[i])
            conf_val = float(signal_conf_arr[i])
            setup_type = int(active_setups_arr[i])

            if sig_long_val > 0 and sig_long_val > sig_short_val:
                raw_signal = sig_long_val
                raw_action = 'long'
            elif sig_short_val > 0:
                raw_signal = -sig_short_val
                raw_action = 'short'
            else:
                raw_signal = 0.0
                raw_action = 'hold'

            if direction_filter == 'bearish' and raw_action == 'long':
                trend_regime = cached_regime.get('trend_regime', 'ranging')
                mult = {'strong_trend': 0.2, 'trending': 0.5, 'ranging': 0.8, 'choppy': 0.85}.get(trend_regime, 0.8)
                raw_signal *= mult
            elif direction_filter == 'bullish' and raw_action == 'short':
                trend_regime = cached_regime.get('trend_regime', 'ranging')
                mult = {'strong_trend': 0.2, 'trending': 0.5, 'ranging': 0.8, 'choppy': 0.85}.get(trend_regime, 0.8)
                raw_signal *= mult

            if raw_signal > signal_min:
                signal = {
                    'action': 'long', 'signal_strength': raw_signal,
                    'confidence': conf_val if raw_signal > 0 else 0,
                    'regime': cached_regime,
                    'price': price, 'atr': float(row.get('atr_14', price * 0.01)),
                }
            elif raw_signal < -signal_min:
                signal = {
                    'action': 'short', 'signal_strength': abs(raw_signal),
                    'confidence': conf_val if raw_signal < 0 else 0,
                    'regime': cached_regime,
                    'price': price, 'atr': float(row.get('atr_14', price * 0.01)),
                }
            else:
                signal = {
                    'action': 'hold', 'signal_strength': 0.0,
                    'confidence': 0.0, 'regime': cached_regime,
                    'price': price, 'atr': float(row.get('atr_14', price * 0.01)),
                }
        else:
            w_start = max(setup_window, i - 45)
            df_window_entry = df_entry.iloc[w_start:i + 1]
            cpu_signal = evaluate_all_setups(df_window_entry, cached_regime)
            signal = cpu_signal

        if open_pos is not None:
            exit_reason = None
            exit_px = price

            if open_pos.side == 'long':
                if price <= open_pos.stop_loss:
                    exit_px = open_pos.stop_loss
                    exit_reason = 'sl'
                elif price >= open_pos.take_profit:
                    exit_px = open_pos.take_profit
                    exit_reason = 'tp'
            else:
                if price >= open_pos.stop_loss:
                    exit_px = open_pos.stop_loss
                    exit_reason = 'sl'
                elif price <= open_pos.take_profit:
                    exit_px = open_pos.take_profit
                    exit_reason = 'tp'

            if exit_reason is None:
                hold_h = (pd.Timestamp(ts) - pd.Timestamp(open_pos.entry_time)).total_seconds() / 3600
                if hold_h > 48:
                    exit_reason = 'time'
                    exit_px = price

            if exit_reason is None:
                action = signal['action']
                if open_pos.side == 'long' and action == 'short':
                    counter_score = signal.get('signal_strength', 0) * max(signal.get('confidence', 0), 0.1)
                    entry_score = abs(open_pos.entry_signal_strength)
                    if counter_score > 2.0 * entry_score and entry_score > 0:
                        exit_reason = 'reversal'
                        exit_px = price
                elif open_pos.side == 'short' and action == 'long':
                    counter_score = signal.get('signal_strength', 0) * max(signal.get('confidence', 0), 0.1)
                    entry_score = abs(open_pos.entry_signal_strength)
                    if counter_score > 2.0 * entry_score and entry_score > 0:
                        exit_reason = 'reversal'
                        exit_px = price

            if exit_reason is not None:
                slip = exit_px * SLIPPAGE
                exit_adj = exit_px + slip if open_pos.side == 'long' else exit_px - slip

                if open_pos.side == 'long':
                    raw_pnl = (exit_adj - open_pos.entry_price) * open_pos.quantity
                else:
                    raw_pnl = (open_pos.entry_price - exit_adj) * open_pos.quantity

                fee = open_pos.position_value * TAKER_FEE
                total_fees += fee
                pnl = raw_pnl - fee

                open_pos.exit_time = ts
                open_pos.exit_price = exit_adj
                open_pos.pnl = pnl
                open_pos.pnl_pct = pnl / equity if equity > 0 else 0
                open_pos.exit_reason = exit_reason
                open_pos.fee_paid += fee

                equity += pnl
                trades.append(open_pos)

                if pnl < 0:
                    consecutive_losses += 1
                else:
                    consecutive_losses = 0

                if consecutive_losses >= 5:
                    risk_per_trade = max(0.005, base_risk_per_trade * 0.5)
                else:
                    risk_per_trade = base_risk_per_trade

                regime_mult = {
                    'strong_trend_bull': 1.0, 'strong_trend_bear': 1.0,
                    'trending_bull': 0.9, 'trending_bear': 0.9,
                    'ranging_low_vol': 0.5, 'ranging_high_vol': 0.6,
                    'default': 0.7,
                }.get(cached_regime.get('composite', 'default'), 0.7)
                risk_per_trade *= regime_mult

                peak_equity = max(peak_equity, equity)
                dd = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0
                max_drawdown_val = max(max_drawdown_val, dd)

                if dd >= MAX_DRAWDOWN_HARD:
                    trading_stopped = True

                open_pos = None

        signal_day = ts.date() if hasattr(ts, 'date') else str(ts)[:10]
        if signal['action'] in ('long', 'short') and open_pos is None and not trading_stopped:
            effective_gate = max(signal_gate, int(equity / initial_capital * signal_gate))
            if daily_signal_counts.get(signal_day, 0) >= effective_gate:
                signal = {'action': 'hold', 'signal_strength': 0.0, 'confidence': 0.0, 'regime': cached_regime, 'price': price, 'atr': float(row.get('atr_14', price * 0.01))}
            elif adx_min > 0 and float(row.get('adx', 0)) < adx_min:
                signal = {'action': 'hold', 'signal_strength': 0.0, 'confidence': 0.0, 'regime': cached_regime, 'price': price, 'atr': float(row.get('atr_14', price * 0.01))}
            else:
                daily_signal_counts[signal_day] = daily_signal_counts.get(signal_day, 0) + 1
            qty = 0.0
            pos_val = 0.0
            lev = 1

            atr_val = row.get('atr_14', price * 0.01)
            pos_info = compute_position_size(equity, price, atr_val, signal,
                {'risk_per_trade': risk_per_trade, 'leverage': leverage, 'sl_mult': sl_mult})
            qty = pos_info['quantity']
            lev = pos_info['leverage']
            pos_val = qty * price

            if pos_val >= equity * 0.02 and pos_val <= equity * leverage and lev >= 1:
                fee = pos_info['position_value'] * TAKER_FEE
                total_fees += fee

                slip = price * SLIPPAGE
                entry_price = price + slip if signal['action'] == 'long' else price - slip

                sl_pct = pos_info['stop_loss_pct']
                tp_pct = pos_info['take_profit_pct']

                if signal['action'] == 'long':
                    sl_price = entry_price * (1 + sl_pct)
                    tp_price = entry_price * (1 + tp_pct)
                else:
                    sl_price = entry_price * (1 - sl_pct)
                    tp_price = entry_price * (1 - tp_pct)

                entry_sig_strength = abs(signal.get('signal_strength', 0.35))
                open_pos = Trade(
                    symbol=symbol,
                    side=signal['action'],
                    entry_time=ts,
                    entry_price=entry_price,
                    quantity=qty,
                    position_value=pos_info['position_value'],
                    leverage=lev,
                    entry_signal_strength=entry_sig_strength,
                    stop_loss=sl_price,
                    take_profit=tp_price,
                    fee_paid=fee,
                )

        equity_curve.append(equity)
        timestamps.append(ts)

    if open_pos is not None:
        last_price = float(recs_entry[-1]['close'])
        if open_pos.side == 'long':
            raw_pnl = (last_price - open_pos.entry_price) * open_pos.quantity
        else:
            raw_pnl = (open_pos.entry_price - last_price) * open_pos.quantity
        fee = open_pos.position_value * TAKER_FEE
        total_fees += fee
        pnl = raw_pnl - fee
        equity += pnl
        open_pos.exit_time = recs_entry[-1]['timestamp']
        open_pos.exit_price = last_price
        prev_eq = equity - pnl
        open_pos.pnl = pnl
        open_pos.pnl_pct = pnl / prev_eq if prev_eq > 0 else 0
        open_pos.exit_reason = 'end'
        open_pos.fee_paid += fee
        trades.append(open_pos)

    return _compute_metrics(symbol, entry_tf, initial_capital, equity, trades,
                            equity_curve, timestamps, total_fees, regime_log)


def _empty_result(symbol, capital, df, entry_tf='1h'):
    ts = df['timestamp'].iloc[0] if len(df) > 0 else pd.Timestamp.now()
    return BacktestResult(symbol=symbol, timeframe=f'{entry_tf}/4h', initial_capital=capital,
                          final_equity=capital, total_pnl=0, total_pnl_pct=0,
                          total_trades=0, winning_trades=0, losing_trades=0, win_rate=0,
                          avg_win=0, avg_loss=0, profit_factor=0, max_drawdown=0,
                          max_drawdown_pct=0, sharpe_ratio=0, sortino_ratio=0,
                          calmar_ratio=0, total_fees=0, avg_hold_bars=0,
                          equity_curve=[capital], timestamps=[ts])


def _compute_metrics(symbol, entry_tf, init_cap, final_eq, trades, eq_curve, timestamps, fees, regime_log):
    total_pnl = final_eq - init_cap
    total_pnl_pct = total_pnl / init_cap * 100 if init_cap > 0 else 0

    winning = [t for t in trades if t.pnl > 0]
    losing = [t for t in trades if t.pnl <= 0]

    win_rate = len(winning) / len(trades) * 100 if trades else 0
    avg_win = float(np.mean([t.pnl for t in winning])) if winning else 0
    avg_loss = float(abs(np.mean([t.pnl for t in losing]))) if losing else 0

    sum_win = sum(t.pnl for t in winning)
    sum_loss = abs(sum(t.pnl for t in losing))
    pf = sum_win / sum_loss if sum_loss > 0 else (99.9 if sum_win > 0 else 0)

    peak = init_cap
    max_dd_val = 0.0
    max_dd_pct = 0.0
    for eq in eq_curve:
        peak = max(peak, eq)
        dd = peak - eq
        dd_pct = dd / peak * 100 if peak > 0 else 0
        max_dd_val = max(max_dd_val, dd)
        max_dd_pct = max(max_dd_pct, dd_pct)

    returns = pd.Series(eq_curve).pct_change().dropna()
    if len(returns) > 10 and float(returns.std()) > 0:
        bars_per_day = {'15m': 96, '30m': 48, '1h': 24}.get(entry_tf, 96)
        ann = np.sqrt(bars_per_day * 365)
        sharpe = float(returns.mean() / returns.std() * ann)
        down = returns[returns < 0]
        sortino = float(returns.mean() / down.std() * ann) if len(down) > 0 and float(down.std()) > 0 else 0
    else:
        sharpe = 0.0
        sortino = 0.0

    calmar = total_pnl_pct / max_dd_pct if max_dd_pct > 0 else 0

    avg_hold = 0
    if trades:
        holds = []
        for t in trades:
            if t.exit_time is not None:
                h = (pd.Timestamp(t.exit_time) - pd.Timestamp(t.entry_time)).total_seconds() / 3600
                holds.append(h)
        avg_hold = float(np.mean(holds)) if holds else 0

    return BacktestResult(
        symbol=symbol, timeframe=f'{entry_tf}/4h', initial_capital=init_cap,
        final_equity=final_eq, total_pnl=total_pnl, total_pnl_pct=total_pnl_pct,
        total_trades=len(trades), winning_trades=len(winning),
        losing_trades=len(losing), win_rate=win_rate, avg_win=avg_win,
        avg_loss=avg_loss, profit_factor=pf, max_drawdown=max_dd_val,
        max_drawdown_pct=max_dd_pct, sharpe_ratio=sharpe, sortino_ratio=sortino,
        calmar_ratio=calmar, total_fees=fees, avg_hold_bars=avg_hold,
        trades=trades, equity_curve=eq_curve, timestamps=timestamps,
        regime_log=regime_log,
    )
