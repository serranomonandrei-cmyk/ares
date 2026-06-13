import numpy as np
import pandas as pd

from .setups import evaluate_all_setups


def generate_combined_signal(df_window_1h, regime):
    return evaluate_all_setups(df_window_1h, regime)


def compute_indicator_confirmation(row_1h: dict, regime: dict) -> float:
    """Indicator-only confirmation multiplier (0.5-1.5).
    Uses 1h indicators to rate the trading context — NOT a trading signal.
    Below 1.0 = poor conditions, above 1.0 = favorable, 1.0 = neutral.
    """
    mult = 1.0

    rsi = row_1h.get('rsi_14', 50)
    if 30 <= rsi <= 70:
        mult *= 1.0
    elif rsi < 20 or rsi > 80:
        mult *= 0.7
    elif rsi < 30:
        mult *= 0.85
    else:
        mult *= 0.85

    bb_pos = row_1h.get('bb_position', 0.5)
    if 0.2 <= bb_pos <= 0.8:
        mult *= 1.0
    elif bb_pos < 0.05 or bb_pos > 0.95:
        mult *= 0.75
    elif bb_pos < 0.15 or bb_pos > 0.85:
        mult *= 0.85

    adx = regime.get('adx', 25)
    if adx > 40:
        mult *= 0.9
    elif adx > 30:
        mult *= 0.95

    macd_hist = row_1h.get('macd_hist', 0)
    macd_hist_roc = row_1h.get('macd_hist_roc', 0)
    if abs(macd_hist) > 0 and macd_hist * macd_hist_roc > 0:
        mult *= 1.1
    elif abs(macd_hist) > 0 and macd_hist * macd_hist_roc < 0:
        mult *= 0.9

    return float(max(0.5, min(1.5, mult)))


def compute_position_size(equity: float, price: float, atr_val: float, signal: dict,
                          config: dict) -> dict:
    risk_per_trade = config.get('risk_per_trade', 0.015)
    max_leverage = config.get('leverage', 10)

    confidence = signal.get('confidence', 0.5)
    atr_pct = atr_val / price if price > 0 else 0.01

    if atr_pct < 0.001:
        atr_pct = 0.01
    atr_percentile = signal.get('regime', {}).get('atr_percentile', 0.5)
    atr_percentile = max(0.1, min(0.9, atr_percentile))
    sl_mult_cfg = config.get('sl_mult', 2.0)
    stop_mult = (sl_mult_cfg - 0.5) + atr_percentile * 1.0
    stop_distance = atr_pct * stop_mult

    vol_regime = signal.get('regime', {}).get('vol_regime', 'normal')
    vol_factor = 0.7 if vol_regime == 'high' else (1.2 if vol_regime == 'low' else 1.0)

    rr_ratio = 3.0
    kelly_win_rate = 0.32
    kelly = kelly_win_rate - (1 - kelly_win_rate) / rr_ratio
    kelly = max(0.01, min(0.25, kelly * 0.25))
    kelly_risk = equity * kelly
    fixed_risk = equity * risk_per_trade * confidence * vol_factor
    risk_amount = min(fixed_risk, kelly_risk)

    raw_position_value = risk_amount / stop_distance if stop_distance > 0 else equity
    max_position_value = equity * max_leverage

    position_value = min(raw_position_value, max_position_value)
    position_value = max(position_value, equity * 0.01)

    leverage_used = min(max_leverage, max(1, position_value / equity))
    quantity = position_value / price if price > 0 else 0
    take_profit_distance = stop_distance * 3.0

    return {
        'quantity': quantity,
        'stop_loss_pct': -stop_distance,
        'take_profit_pct': take_profit_distance,
        'position_value': position_value,
        'leverage': leverage_used,
        'risk_amount': risk_amount,
    }
