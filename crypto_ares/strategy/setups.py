import numpy as np
import pandas as pd


def detect_swing_points(highs, lows, lookback=10):
    n = len(highs)
    swing_high_idx = None
    swing_low_idx = None
    swing_high_val = None
    swing_low_val = None

    for i in range(max(1, n - lookback), n - 1):
        if highs[i] > highs[i - 1] and highs[i] > highs[i + 1]:
            if swing_high_idx is None or highs[i] > swing_high_val:
                swing_high_idx = i
                swing_high_val = highs[i]
        if lows[i] < lows[i - 1] and lows[i] < lows[i + 1]:
            if swing_low_idx is None or lows[i] < swing_low_val:
                swing_low_idx = i
                swing_low_val = lows[i]

    return swing_high_idx, swing_high_val, swing_low_idx, swing_low_val


def detect_swing_points_all(highs, lows, lookback=20):
    """Find all swing highs and lows in the window."""
    swing_highs = []
    swing_lows = []
    for i in range(1, len(highs) - 1):
        if highs[i] > highs[i - 1] and highs[i] > highs[i + 1]:
            swing_highs.append((i, highs[i]))
        if lows[i] < lows[i - 1] and lows[i] < lows[i + 1]:
            swing_lows.append((i, lows[i]))
    return swing_highs, swing_lows


def liquidity_sweep_bos(df: pd.DataFrame) -> dict:
    """
    Liquidity Sweep + Break of Structure.
    1. Price sweeps below recent swing low (long) / above recent swing high (short)
    2. Then reverses and breaks the previous swing in opposite direction
    3. Entry on retest or confirmation candle
    """
    if len(df) < 15:
        return {'signal': 0, 'confidence': 0}

    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values
    opens = df['open'].values

    sh_idx, sh_val, sl_idx, sl_val = detect_swing_points(highs, lows, 12)

    if sh_idx is None or sl_idx is None:
        return {'signal': 0, 'confidence': 0}

    c = closes[-1]
    prev_close = closes[-2] if len(closes) >= 2 else c
    prev_high = highs[-2] if len(highs) >= 2 else highs[-1]
    prev_low = lows[-2] if len(lows) >= 2 else lows[-1]

    result = {'signal': 0, 'confidence': 0}

    # Bullish: sweep low, then break structure
    bars_since_swing_low = len(highs) - 1 - sl_idx if sl_idx is not None else 99
    sweep_occurred = any(lows[j] < sl_val for j in range(sl_idx, len(lows))) if sl_val else False
    recent_sweep = any(
        closes[j] < sl_val for j in range(max(sl_idx, len(highs) - 5), len(highs))
    ) if sl_val else False

    if sweep_occurred and recent_sweep and sl_idx < len(highs) - 3:
        bos_level = sl_val
        if closes[-1] > max(highs[max(sl_idx, len(highs)-8):-1]) and closes[-1] > closes[-2]:
            if prev_close < closes[-1]:
                result['signal'] = 1
                result['confidence'] = 0.6 + min(0.3, bars_since_swing_low / 20)
                result['setup'] = 'lq_sweep_bos_long'
                result['entry_level'] = closes[-1]
                result['stop_level'] = min(lows[-3:]) * 0.995 if len(lows) >= 3 else closes[-1] * 0.99
                return result

    # Bearish: sweep high, then break structure
    bars_since_swing_high = len(highs) - 1 - sh_idx if sh_idx is not None else 99
    sweep_occurred = any(highs[j] > sh_val for j in range(sh_idx, len(highs))) if sh_val else False
    recent_sweep = any(
        closes[j] > sh_val for j in range(max(sh_idx, len(highs) - 5), len(highs))
    ) if sh_val else False

    if sweep_occurred and recent_sweep and sh_idx < len(highs) - 3:
        if closes[-1] < min(lows[max(sh_idx, len(highs)-8):-1]) and closes[-1] < closes[-2]:
            if prev_close > opens[-1]:
                result['signal'] = -1
                result['confidence'] = 0.6 + min(0.3, bars_since_swing_high / 20)
                result['setup'] = 'lq_sweep_bos_short'
                result['entry_level'] = closes[-1]
                result['stop_level'] = max(highs[-3:]) * 1.005 if len(highs) >= 3 else closes[-1] * 1.01
                return result

    return result


def fair_value_gap(df: pd.DataFrame) -> dict:
    """
    Fair Value Gap (imbalance) detection.
    Bullish FVG: candle 2 low > candle 1 high (gap up), price returns to fill
    Bearish FVG: candle 2 high < candle 1 low (gap down), price returns to fill
    Entry on touch of gap zone.
    """
    if len(df) < 6:
        return {'signal': 0, 'confidence': 0}

    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values
    opens = df['open'].values

    result = {'signal': 0, 'confidence': 0}

    for i in range(1, min(6, len(df) - 2)):
        c1_h, c1_l = highs[i-1], lows[i-1]
        c2_h, c2_l = highs[i], lows[i]
        c3_h, c3_l = highs[i+1], lows[i+1]

        # Bullish FVG: gap up between c1 and c3
        if c3_l > c1_h and c2_l > c1_h:
            gap_top = c3_l
            gap_bot = c1_h
            gap_size = gap_top - gap_bot
            if gap_size > 0 and closes[-1] >= gap_bot and closes[-1] <= gap_top:
                touch_depth = (gap_top - closes[-1]) / gap_size
                result['signal'] = 1
                result['confidence'] = 0.5 + min(0.3, touch_depth * 0.5)
                result['setup'] = 'fvg_bullish'
                return result

        # Bearish FVG: gap down between c1 and c3
        if c3_h < c1_l and c2_h < c1_l:
            gap_top = c1_l
            gap_bot = c3_h
            gap_size = gap_top - gap_bot
            if gap_size > 0 and closes[-1] >= gap_bot and closes[-1] <= gap_top:
                touch_depth = (closes[-1] - gap_bot) / gap_size
                result['signal'] = -1
                result['confidence'] = 0.5 + min(0.3, touch_depth * 0.5)
                result['setup'] = 'fvg_bearish'
                return result

    return result


def swing_failure_pattern(df: pd.DataFrame) -> dict:
    """
    Swing Failure Pattern (SFP) / Trap.
    Bullish: price makes new low below recent swing, immediately closes back above
    Bearish: price makes new high above recent swing, immediately closes back below
    """
    if len(df) < 12:
        return {'signal': 0, 'confidence': 0}

    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values
    opens = df['open'].values

    _, _, sl_idx, sl_val = detect_swing_points(highs, lows, 10)
    sh_idx, sh_val, _, _ = detect_swing_points(highs, lows, 10)

    result = {'signal': 0, 'confidence': 0}

    # Bullish SFP: new low below last swing low, close back above
    if sl_val is not None and sl_idx < len(lows) - 2:
        recent_lows = lows[max(sl_idx, len(lows)-7):]
        recent_closes = closes[max(sl_idx, len(lows)-7):]
        if recent_lows[-1] < sl_val and closes[-1] > sl_val and closes[-1] > opens[-1]:
            penetration = (sl_val - recent_lows[-1]) / (sl_val * 0.01 + 1e-10)
            result['signal'] = 1
            result['confidence'] = min(0.85, 0.5 + min(0.35, penetration / 2))
            result['setup'] = 'sfp_bullish'
            return result

    # Bearish SFP: new high above last swing high, close back below
    if sh_val is not None and sh_idx < len(highs) - 2:
        recent_highs = highs[max(sh_idx, len(highs)-7):]
        recent_closes = closes[max(sh_idx, len(highs)-7):]
        if recent_highs[-1] > sh_val and closes[-1] < sh_val and closes[-1] < opens[-1]:
            penetration = (recent_highs[-1] - sh_val) / (sh_val * 0.01 + 1e-10)
            result['signal'] = -1
            result['confidence'] = min(0.85, 0.5 + min(0.35, penetration / 2))
            result['setup'] = 'sfp_bearish'
            return result

    return result


def detect_divergence(df: pd.DataFrame) -> dict:
    """
    RSI divergence detection.
    Regular Bullish: price lower low, RSI higher low
    Regular Bearish: price higher high, RSI lower high
    Hidden Bullish: price higher low, RSI lower low (continuation)
    Hidden Bearish: price lower high, RSI higher high (continuation)
    """
    if len(df) < 20:
        return {'signal': 0, 'confidence': 0}

    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values
    rsi = df['rsi_14'].values if 'rsi_14' in df.columns else None

    if rsi is None or len(rsi) < 20:
        return {'signal': 0, 'confidence': 0}

    sh_idx, sh_val, sl_idx, sl_val = detect_swing_points(highs, lows, 14)
    rsi_sh_idx, _, rsi_sl_idx, _ = detect_swing_points(rsi, rsi, 14)

    result = {'signal': 0, 'confidence': 0}

    # Regular Bullish Divergence
    if sl_idx is not None and rsi_sl_idx is not None:
        if sl_idx > rsi_sl_idx + 2:
            price_low = lows[sl_idx]
            rsi_low_current = rsi[sl_idx]
            rsi_low_prev = rsi[rsi_sl_idx]
            price_low_prev = lows[rsi_sl_idx]
            if price_low < price_low_prev and rsi_low_current > rsi_low_prev:
                if closes[-1] > lows[sl_idx] * 1.002:
                    result['signal'] = 1
                    result['confidence'] = 0.55 + min(0.3, (rsi_low_current - rsi_low_prev) * 2)
                    result['setup'] = 'div_bullish'
                    return result

    # Regular Bearish Divergence
    if sh_idx is not None and rsi_sh_idx is not None:
        if sh_idx > rsi_sh_idx + 2:
            price_high = highs[sh_idx]
            rsi_high_current = rsi[sh_idx]
            rsi_high_prev = rsi[rsi_sh_idx]
            price_high_prev = highs[rsi_sh_idx]
            if price_high > price_high_prev and rsi_high_current < rsi_high_prev:
                if closes[-1] < highs[sh_idx] * 0.998:
                    result['signal'] = -1
                    result['confidence'] = 0.55 + min(0.3, (rsi_high_prev - rsi_high_current) * 2)
                    result['setup'] = 'div_bearish'
                    return result

    return result


def engulfing_at_key_level(df: pd.DataFrame) -> dict:
    """
    Bullish/Bearish Engulfing pattern at a key level (EMA, swing point).
    Bullish: current candle open < prev close, close > prev open (full body engulf up)
    Bearish: current candle open > prev close, close < prev open (full body engulf down)
    Also checks proximity to EMA or recent swing level.
    """
    if len(df) < 6:
        return {'signal': 0, 'confidence': 0}

    opens = df['open'].values
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values
    ema_50 = df['ema_50'].values if 'ema_50' in df.columns else None
    ema_200 = df['ema_200'].values if 'ema_200' in df.columns else None

    result = {'signal': 0, 'confidence': 0}

    def near_level(price, ema_val, threshold=0.005):
        return ema_val is not None and abs(price / ema_val - 1) < threshold

    # Need at least 2 candles
    if len(closes) < 2:
        return result

    c0_close = closes[-2]
    c0_open = opens[-2]
    c1_open = opens[-1]
    c1_close = closes[-1]
    c1_high = highs[-1]
    c1_low = lows[-1]

    # Need previous candle body
    prev_body = abs(c0_close - c0_open)
    if prev_body == 0:
        return result

    # Bullish Engulfing
    if (c1_open < c0_close and c1_close > c0_open):
        body_ratio = abs(c1_close - c1_open) / prev_body
        if body_ratio > 0.6:
            level_bonus = 0
            if ema_50 is not None and near_level(c1_low, ema_50[-1]):
                level_bonus = 0.2
            if ema_200 is not None and near_level(c1_low, ema_200[-1]):
                level_bonus = 0.3
            result['signal'] = 1
            result['confidence'] = 0.5 + min(0.3, body_ratio * 0.2) + level_bonus
            result['setup'] = 'engulf_bullish'
            return result

    # Bearish Engulfing
    if (c1_open > c0_close and c1_close < c0_open):
        body_ratio = abs(c1_close - c1_open) / prev_body
        if body_ratio > 0.6:
            level_bonus = 0
            if ema_50 is not None and near_level(c1_high, ema_50[-1]):
                level_bonus = 0.2
            if ema_200 is not None and near_level(c1_high, ema_200[-1]):
                level_bonus = 0.3
            result['signal'] = -1
            result['confidence'] = 0.5 + min(0.3, body_ratio * 0.2) + level_bonus
            result['setup'] = 'engulf_bearish'
            return result

    return result


def inside_bar_breakout(df: pd.DataFrame) -> dict:
    """
    Inside Bar Breakout.
    Current range inside previous candle range.
    Breakout above mother bar high = long, below mother bar low = short.
    Requires volume confirmation.
    """
    if len(df) < 4:
        return {'signal': 0, 'confidence': 0}

    opens = df['open'].values
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values
    volumes = df['volume'].values if 'volume' in df.columns else None
    vol_sma = df['volume_sma_24'].values if 'volume_sma_24' in df.columns else None

    result = {'signal': 0, 'confidence': 0}

    if len(highs) < 3:
        return result

    # Check if current bar is an inside bar
    mom_high = highs[-2]
    mom_low = lows[-2]
    cur_high = highs[-1]
    cur_low = lows[-1]
    cur_close = closes[-1]
    cur_volume = volumes[-1] if volumes is not None else 1
    avg_vol = vol_sma[-1] if vol_sma is not None else cur_volume

    is_inside = cur_high <= mom_high * 1.001 and cur_low >= mom_low * 0.999
    inside_body = abs(closes[-1] - opens[-1]) < abs(closes[-2] - opens[-2]) * 1.1

    if not (is_inside or inside_body):
        return result

    vol_ratio = cur_volume / avg_vol if avg_vol > 0 else 1

    # Bullish breakout: close above mother high
    if cur_close > mom_high and vol_ratio > 1.2:
        result['signal'] = 1
        result['confidence'] = 0.5 + min(0.3, (vol_ratio - 1) * 0.3) + min(0.15, (cur_close - mom_high) / (mom_high * 0.01 + 1e-10) * 0.05)
        result['setup'] = 'ib_breakout_long'
        return result

    # Bearish breakout: close below mother low
    if cur_close < mom_low and vol_ratio > 1.2:
        result['signal'] = -1
        result['confidence'] = 0.5 + min(0.3, (vol_ratio - 1) * 0.3) + min(0.15, (mom_low - cur_close) / (mom_low * 0.01 + 1e-10) * 0.05)
        result['setup'] = 'ib_breakout_short'
        return result

    return result


def structure_shift(df: pd.DataFrame) -> dict:
    """
    Market Structure Shift (MSS) / Change of Character.
    Detects break of trendline or structure level with retest.
    Bullish: breaks above recent swing high (BOS), retests broken level
    Bearish: breaks below recent swing low (BOS), retests broken level
    """
    if len(df) < 20:
        return {'signal': 0, 'confidence': 0}

    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values

    swing_highs, swing_lows = detect_swing_points_all(highs, lows, 15)

    result = {'signal': 0, 'confidence': 0}

    if len(swing_lows) >= 2:
        last_ll = swing_lows[-1]
        prev_ll = swing_lows[-2]
        if last_ll[1] > prev_ll[1]:
            level = last_ll[1]
            last_swing_idx = last_ll[0]
            retest_idx = last_swing_idx + 2
            if retest_idx < len(closes):
                retest_range = closes[retest_idx:]
                for retest_i, retest_price in enumerate(retest_range):
                    if abs(retest_price / level - 1) < 0.005:
                        if retest_price > level:
                            result['signal'] = 1
                            result['confidence'] = 0.55
                            result['setup'] = 'mss_bullish'
                            return result

    if len(swing_highs) >= 2:
        last_hh = swing_highs[-1]
        prev_hh = swing_highs[-2]
        if last_hh[1] < prev_hh[1]:
            level = last_hh[1]
            last_swing_idx = last_hh[0]
            retest_idx = last_swing_idx + 2
            if retest_idx < len(closes):
                retest_range = closes[retest_idx:]
                for retest_i, retest_price in enumerate(retest_range):
                    if abs(retest_price / level - 1) < 0.005:
                        if retest_price < level:
                            result['signal'] = -1
                            result['confidence'] = 0.55
                            result['setup'] = 'mss_bearish'
                            return result

    return result


def evaluate_all_setups(df_window: pd.DataFrame, regime: dict) -> dict:
    """Run all setup detectors and produce a combined ensemble signal."""
    detectors = [
        ('liquidity_sweep_bos', liquidity_sweep_bos, 0.25),
        ('fair_value_gap', fair_value_gap, 0.15),
        ('swing_failure', swing_failure_pattern, 0.15),
        ('divergence', detect_divergence, 0.15),
        ('engulfing', engulfing_at_key_level, 0.10),
        ('inside_bar', inside_bar_breakout, 0.10),
        ('structure_shift', structure_shift, 0.10),
    ]

    direction = regime.get('direction', 'neutral')
    trend_regime = regime.get('trend_regime', 'ranging')
    vol_regime = regime.get('vol_regime', 'normal')

    results = {}
    weighted_signal = 0.0
    total_weight = 0.0
    active_setups = []

    for name, detector, base_weight in detectors:
        adj_weight = base_weight

        # Regime-based weight adjustments
        if name == 'liquidity_sweep_bos':
            if trend_regime in ('trending', 'strong_trend'):
                adj_weight *= 1.3
            if vol_regime == 'high':
                adj_weight *= 1.2
        elif name == 'fair_value_gap':
            if vol_regime == 'low':
                adj_weight *= 1.4
        elif name == 'swing_failure':
            if vol_regime == 'high':
                adj_weight *= 1.3
        elif name == 'divergence':
            if trend_regime == 'trending':
                adj_weight *= 1.2
        elif name == 'engulfing':
            if vol_regime in ('low', 'normal'):
                adj_weight *= 1.2
        elif name == 'inside_bar':
            if trend_regime in ('trending', 'strong_trend'):
                adj_weight *= 1.3
        elif name == 'structure_shift':
            if trend_regime in ('ranging',):
                adj_weight *= 1.3

        try:
            det_result = detector(df_window)
        except Exception:
            det_result = {'signal': 0, 'confidence': 0}

        sig = det_result.get('signal', 0)
        conf = det_result.get('confidence', 0)

        # Direction filter
        if sig > 0 and direction == 'bearish':
            adj_weight *= 0.4
        elif sig < 0 and direction == 'bullish':
            adj_weight *= 0.4

        result_entry = {
            'signal': sig,
            'confidence': conf,
            'weight': adj_weight,
            'setup': det_result.get('setup', ''),
        }
        results[name] = result_entry

        contribution = sig * conf * adj_weight
        weighted_signal += contribution
        total_weight += adj_weight

        if abs(sig) > 0 and conf > 0.3:
            active_setups.append(det_result.get('setup', name))

    if total_weight > 0:
        final_signal = weighted_signal / total_weight
    else:
        final_signal = 0.0

    if final_signal > 0.15:
        action = 'long'
    elif final_signal < -0.15:
        action = 'short'
    else:
        action = 'hold'

    confidence = min(1.0, abs(final_signal))
    price = df_window['close'].iloc[-1]
    atr_val = df_window['atr_14'].iloc[-1] if 'atr_14' in df_window.columns else price * 0.01

    return {
        'action': action,
        'signal_strength': float(final_signal),
        'confidence': float(confidence),
        'setup_results': results,
        'active_setups': active_setups,
        'price': float(price),
        'atr': float(atr_val),
    }
