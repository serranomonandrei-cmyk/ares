import numpy as np
import pandas as pd

from ..data.features import ema, sma, rsi, atr, macd, bollinger_bands, adx

def trend_continuation_signal(df: pd.DataFrame, regime: dict) -> float:
    c = df['close']
    h = df['high']
    l = df['low']
    v = df['volume']

    ema_9 = ema(c, 9)
    ema_21 = ema(c, 21)
    ema_50 = ema(c, 50)
    adx_val, plus_di, minus_di = adx(h, l, c, 14)
    macd_line, signal_line, hist = macd(c, 12, 26, 9)

    latest = {
        'ema_9': ema_9.iloc[-1],
        'ema_21': ema_21.iloc[-1],
        'ema_50': ema_50.iloc[-1],
        'adx': adx_val.iloc[-1],
        'plus_di': plus_di.iloc[-1],
        'minus_di': minus_di.iloc[-1],
        'macd_line': macd_line.iloc[-1],
        'signal_line': signal_line.iloc[-1],
        'hist': hist.iloc[-1],
        'hist_prev': hist.iloc[-2] if len(hist) >= 2 else 0,
        'close': c.iloc[-1],
        'volume': v.iloc[-1],
        'volume_sma': v.rolling(24).mean().iloc[-1],
    }

    signal = 0.0
    reasons = []

    long_cond = (
        latest['close'] > ema_50.iloc[-1]
        and ema_9.iloc[-1] > ema_21.iloc[-1]
        and latest['adx'] > 22
        and latest['macd_line'] > latest['signal_line']
        and latest['hist'] > latest['hist_prev']
    )
    if long_cond:
        base = 0.6
        adx_bonus = min(0.3, (latest['adx'] - 22) / 60)
        vol_conf = min(0.1, latest['volume'] / latest['volume_sma'] * 0.05) if latest['volume_sma'] > 0 else 0
        signal = base + adx_bonus + vol_conf
        reasons.append('trend_long')

    short_cond = (
        latest['close'] < ema_50.iloc[-1]
        and ema_9.iloc[-1] < ema_21.iloc[-1]
        and latest['adx'] > 22
        and latest['macd_line'] < latest['signal_line']
        and latest['hist'] < latest['hist_prev']
    )
    if short_cond:
        base = -0.6
        adx_bonus = min(0.3, (latest['adx'] - 22) / 60)
        vol_conf = min(0.1, latest['volume'] / latest['volume_sma'] * 0.05) if latest['volume_sma'] > 0 else 0
        signal = -(abs(base) + adx_bonus + vol_conf)
        reasons.append('trend_short')

    direction = regime.get('direction', 'neutral')
    if signal > 0 and direction == 'bearish':
        signal *= 0.5
    elif signal < 0 and direction == 'bullish':
        signal *= 0.5

    if abs(signal) < 0.3:
        signal = 0.0

    return signal


def mean_reversion_signal(df: pd.DataFrame, regime: dict) -> float:
    c = df['close']
    h = df['high']
    l = df['low']

    rsi_14 = rsi(c, 14)
    bb_up, bb_mid, bb_low = bollinger_bands(c, 20, 2.0)
    stoch_k, stoch_d = stochastic_cc(h, l, c, 14, 3)
    atr_val = atr(h, l, c, 14)

    latest = {
        'rsi': rsi_14.iloc[-1],
        'close': c.iloc[-1],
        'bb_up': bb_up.iloc[-1],
        'bb_low': bb_low.iloc[-1],
        'bb_mid': bb_mid.iloc[-1],
        'stoch_k': stoch_k.iloc[-1],
        'stoch_d': stoch_d.iloc[-1],
        'atr': atr_val.iloc[-1],
    }

    vol_regime = regime.get('vol_regime', 'normal')
    vol_adj = 0.7 if vol_regime == 'high' else 1.0

    signal = 0.0
    reasons = []

    rsi_oversold = rsi_14.iloc[-2] if len(rsi_14) >= 2 else rsi_14.iloc[-1]
    rsi_overbought = rsi_14.iloc[-2] if len(rsi_14) >= 2 else rsi_14.iloc[-1]

    if latest['rsi'] < 30 and latest['close'] <= latest['bb_low'] * 1.005:
        strength = (30 - latest['rsi']) / 30
        bb_bounce = (latest['bb_low'] - latest['close']) / latest['atr']
        stoch_conf = 1 if latest['stoch_k'] < 20 and latest['stoch_d'] < 20 else 0.5
        signal = min(0.9, (strength * 0.4 + min(1, abs(bb_bounce)) * 0.3 + stoch_conf * 0.3)) * vol_adj
        reasons.append('mr_long')
    elif latest['rsi'] > 70 and latest['close'] >= latest['bb_up'] * 0.995:
        strength = (latest['rsi'] - 70) / 30
        bb_reject = (latest['close'] - latest['bb_up']) / latest['atr']
        stoch_conf = 1 if latest['stoch_k'] > 80 and latest['stoch_d'] > 80 else 0.5
        signal = -min(0.9, (strength * 0.4 + min(1, abs(bb_reject)) * 0.3 + stoch_conf * 0.3)) * vol_adj
        reasons.append('mr_short')

    if abs(signal) < 0.25:
        signal = 0.0

    return signal


def breakout_signal(df: pd.DataFrame, regime: dict) -> float:
    c = df['close']
    h = df['high']
    l = df['low']
    v = df['volume']

    atr_val = atr(h, l, c, 14)
    highest_20 = h.rolling(20).max()
    lowest_20 = l.rolling(20).min()
    vol_sma_24 = v.rolling(24).mean()
    vol_ratio = v / vol_sma_24.replace(0, np.nan)

    dc_width = (highest_20 - lowest_20) / lowest_20 * 100

    latest = {
        'close': c.iloc[-1],
        'high_20': highest_20.iloc[-1],
        'low_20': lowest_20.iloc[-1],
        'atr': atr_val.iloc[-1],
        'vol_ratio': vol_ratio.iloc[-1],
        'dc_width': dc_width.iloc[-1],
        'volume': v.iloc[-1],
        'vol_sma_24': vol_sma_24.iloc[-1],
    }

    prev_high_20 = highest_20.iloc[-2] if len(highest_20) >= 2 else latest['high_20']
    prev_low_20 = lowest_20.iloc[-2] if len(lowest_20) >= 2 else latest['low_20']

    signal = 0.0
    reasons = []

    long_cond = (
        latest['close'] > latest['high_20']
        and latest['close'] > prev_high_20
        and latest['vol_ratio'] > 1.3
        and latest['dc_width'] > 2.0
    )
    if long_cond:
        vol_strength = min(1.0, (latest['vol_ratio'] - 1) / 2)
        bko_strength = (latest['close'] - latest['high_20']) / latest['atr']
        bko_strength = min(1.0, max(0, bko_strength))
        signal = 0.5 + vol_strength * 0.25 + bko_strength * 0.25
        reasons.append('breakout_long')

    short_cond = (
        latest['close'] < latest['low_20']
        and latest['close'] < prev_low_20
        and latest['vol_ratio'] > 1.3
        and latest['dc_width'] > 2.0
    )
    if short_cond:
        vol_strength = min(1.0, (latest['vol_ratio'] - 1) / 2)
        bko_strength = (latest['low_20'] - latest['close']) / latest['atr']
        bko_strength = min(1.0, max(0, bko_strength))
        signal = -(0.5 + vol_strength * 0.25 + bko_strength * 0.25)
        reasons.append('breakout_short')

    if abs(signal) < 0.35:
        signal = 0.0

    return signal


def pullback_signal(df: pd.DataFrame, regime: dict) -> float:
    c = df['close']
    h = df['high']
    l = df['low']

    ema_9 = ema(c, 9)
    ema_21 = ema(c, 21)
    ema_50 = ema(c, 50)
    ema_200 = ema(c, 200)
    rsi_14 = rsi(c, 14)
    atr_val = atr(h, l, c, 14)
    macd_line, signal_line, hist = macd(c, 12, 26, 9)

    latest = {
        'close': c.iloc[-1],
        'ema_9': ema_9.iloc[-1],
        'ema_21': ema_21.iloc[-1],
        'ema_50': ema_50.iloc[-1],
        'ema_200': ema_200.iloc[-1],
        'rsi': rsi_14.iloc[-1],
        'atr': atr_val.iloc[-1],
        'macd_line': macd_line.iloc[-1],
        'signal_line': signal_line.iloc[-1],
    }

    direction = regime.get('direction', 'neutral')

    signal = 0.0
    reasons = []

    if direction == 'bullish':
        pullback_to_ema = (latest['close'] - latest['ema_50']) / latest['atr']
        if (
            latest['close'] > latest['ema_200']
            and latest['close'] < latest['ema_50'] * 1.03
            and latest['rsi'] >= 35
            and latest['rsi'] <= 55
            and pullback_to_ema < 0
            and abs(pullback_to_ema) < 2.0
            and latest['close'] > latest['ema_9']
        ):
            depth = min(1.0, abs(pullback_to_ema) / 2.0)
            rsi_recovery = (latest['rsi'] - 35) / 20
            signal = 0.4 + depth * 0.3 + min(0.3, rsi_recovery * 0.3)
            reasons.append('pullback_long')

    elif direction == 'bearish':
        pullback_to_ema = (latest['close'] - latest['ema_50']) / latest['atr']
        if (
            latest['close'] < latest['ema_200']
            and latest['close'] > latest['ema_50'] * 0.97
            and latest['rsi'] >= 45
            and latest['rsi'] <= 65
            and pullback_to_ema > 0
            and pullback_to_ema < 2.0
            and latest['close'] < latest['ema_9']
        ):
            depth = min(1.0, pullback_to_ema / 2.0)
            rsi_exhaustion = (65 - latest['rsi']) / 20
            signal = -(0.4 + depth * 0.3 + min(0.3, rsi_exhaustion * 0.3))
            reasons.append('pullback_short')

    if abs(signal) < 0.3:
        signal = 0.0

    return signal


def stochastic_cc(high, low, close, k_period=14, d_period=3):
    low_min = low.rolling(k_period).min()
    high_max = high.rolling(k_period).max()
    k = 100 * (close - low_min) / (high_max - low_min + 1e-10)
    d = k.rolling(d_period).mean()
    return k, d
