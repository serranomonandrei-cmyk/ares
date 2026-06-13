import numpy as np
import pandas as pd

from ..data.features import sma, ema, macd, atr

def detect_regime(df_4h: pd.DataFrame) -> dict:
    if df_4h.empty or len(df_4h) < 200:
        return {'regime': 'unknown', 'vol_regime': 'normal', 'trend_regime': 'ranging',
                'direction': 'neutral', 'composite': 'default'}

    c, h, l = df_4h['close'], df_4h['high'], df_4h['low']
    atr_val = atr(h, l, c, 14)
    atr_pct = atr_val / c * 100
    atr_percentile = atr_pct.rank(pct=True).iloc[-1]

    if atr_percentile < 0.3:
        vol_regime = 'low'
    elif atr_percentile > 0.7:
        vol_regime = 'high'
    else:
        vol_regime = 'normal'

    df_4h = df_4h.copy()
    from ..data.features import adx as compute_adx
    adx_val, plus_di, minus_di = compute_adx(h, l, c, 14)
    latest_adx = adx_val.iloc[-1]
    latest_plus_di = plus_di.iloc[-1]
    latest_minus_di = minus_di.iloc[-1]

    if latest_adx < 20:
        trend_regime = 'ranging'
    elif latest_adx < 40:
        trend_regime = 'trending'
    else:
        trend_regime = 'strong_trend'

    ema_50 = ema(c, 50)
    ema_200 = ema(c, 200)
    ema_slope = (ema_50.iloc[-1] - ema_50.iloc[-10]) / ema_50.iloc[-10]

    macd_line, signal_line, hist = macd(c, 12, 26, 9)
    macd_bull = macd_line.iloc[-1] > signal_line.iloc[-1]
    macd_hist_pos = hist.iloc[-1] > hist.iloc[-5] if len(hist) >= 5 else True

    price_vs_200 = c.iloc[-1] / ema_200.iloc[-1] - 1
    slope_50 = ema_slope
    slope_9 = (ema(c, 9).iloc[-1] - ema(c, 9).iloc[-3]) / ema(c, 9).iloc[-3]

    if slope_50 > 0.005 and macd_bull:
        direction = 'bullish'
    elif slope_50 < -0.005 and not macd_bull:
        direction = 'bearish'
    else:
        direction = 'neutral'

    cum_return = (c.iloc[-1] / c.iloc[-50] - 1) if len(c) >= 50 else 0
    rolling_max = c.rolling(50).max().iloc[-1]
    drawdown = (c.iloc[-1] / rolling_max - 1) if rolling_max != 0 else 0

    if price_vs_200 > 0.02 and cum_return > 0.05 and direction == 'bullish':
        phase = 'markup'
    elif price_vs_200 > 0.02 and cum_return < -0.03:
        phase = 'distribution'
    elif price_vs_200 < -0.02 and cum_return < -0.05 and direction == 'bearish':
        phase = 'markdown'
    elif price_vs_200 < -0.02 and cum_return > 0.03:
        phase = 'accumulation'
    elif price_vs_200 > 0:
        phase = 'markup' if direction == 'bullish' else 'distribution'
    else:
        phase = 'markdown' if direction == 'bearish' else 'accumulation'

    if trend_regime in ('strong_trend', 'trending') and direction in ('bullish', 'bearish'):
        composite = f"{trend_regime}_{direction}"
    elif trend_regime == 'ranging' and vol_regime in ('low', 'normal'):
        composite = f"ranging_low_vol"
    elif trend_regime == 'ranging' and vol_regime == 'high':
        composite = "ranging_high_vol"
    elif direction == 'neutral':
        composite = 'default'
    else:
        composite = f"trending_{direction}" if direction != 'neutral' else 'default'

    return {
        'vol_regime': vol_regime,
        'trend_regime': trend_regime,
        'direction': direction,
        'phase': phase,
        'composite': composite,
        'adx': latest_adx,
        'plus_di': latest_plus_di,
        'minus_di': latest_minus_di,
        'atr_percentile': atr_percentile,
        'ema_slope_50': slope_50,
        'price_vs_ema200': price_vs_200,
        'cum_return_50': cum_return,
        'drawdown_50': drawdown,
    }

def regime_to_weights(regime: dict) -> dict:
    composite = regime.get('composite', 'default')

    weight_map = {
        'trending_bullish': {'trend': 0.40, 'mean_reversion': 0.10, 'breakout': 0.25, 'pullback': 0.25},
        'trending_bearish': {'trend': 0.40, 'mean_reversion': 0.10, 'breakout': 0.25, 'pullback': 0.25},
        'strong_trend_bullish': {'trend': 0.50, 'mean_reversion': 0.05, 'breakout': 0.30, 'pullback': 0.15},
        'strong_trend_bearish': {'trend': 0.50, 'mean_reversion': 0.05, 'breakout': 0.30, 'pullback': 0.15},
        'ranging_low_vol': {'trend': 0.10, 'mean_reversion': 0.45, 'breakout': 0.20, 'pullback': 0.25},
        'ranging_high_vol': {'trend': 0.15, 'mean_reversion': 0.20, 'breakout': 0.45, 'pullback': 0.20},
    }

    vol = regime.get('vol_regime', 'normal')
    if composite in weight_map:
        weights = weight_map[composite]
    elif vol == 'low':
        weights = weight_map['ranging_low_vol']
    elif vol == 'high':
        weights = weight_map['ranging_high_vol']
    else:
        weights = {'trend': 0.25, 'mean_reversion': 0.25, 'breakout': 0.25, 'pullback': 0.25}

    return weights
