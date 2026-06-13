import pandas as pd
import numpy as np

def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()

def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period).mean()

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def bollinger_bands(series: pd.Series, period: int = 20, std_dev: float = 2.0):
    middle = sma(series, period)
    std = series.rolling(period).std(ddof=0)
    upper = middle + std_dev * std
    lower = middle - std_dev * std
    return upper, middle, lower

def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    tr = atr(high, low, close, period)
    up_move = high - high.shift()
    down_move = low.shift() - low
    plus_dm = ((up_move > down_move) & (up_move > 0)).astype(float) * up_move
    minus_dm = ((down_move > up_move) & (down_move > 0)).astype(float) * down_move
    plus_di = 100 * ema(plus_dm, period) / tr
    minus_di = 100 * ema(minus_dm, period) / tr
    dx = ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)) * 100
    return ema(dx, period), plus_di, minus_di

def stochastic(high: pd.Series, low: pd.Series, close: pd.Series, k_period: int = 14, d_period: int = 3):
    low_min = low.rolling(k_period).min()
    high_max = high.rolling(k_period).max()
    k = 100 * (close - low_min) / (high_max - low_min + 1e-10)
    d = sma(k, d_period)
    return k, d

def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff()).fillna(0)
    return (direction * volume).cumsum()

def money_flow_index(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, period: int = 14) -> pd.Series:
    typical = (high + low + close) / 3
    mf = typical * volume
    direction = (typical > typical.shift()).astype(int) * 2 - 1
    mf_signed = mf * direction.replace(0, np.nan)
    mf_positive = mf_signed.clip(lower=0).rolling(period).sum()
    mf_negative = (-mf_signed).clip(lower=0).rolling(period).sum()
    mfr = mf_positive / mf_negative.replace(0, np.nan)
    return 100 - (100 / (1 + mfr))

def ichimoku(high: pd.Series, low: pd.Series, close: pd.Series):
    tenkan = (high.rolling(9).max() + low.rolling(9).min()) / 2
    kijun = (high.rolling(26).max() + low.rolling(26).min()) / 2
    senkou_a = (tenkan + kijun) / 2
    senkou_b = (high.rolling(52).max() + low.rolling(52).min()) / 2
    chikou = close.shift(-26)
    return tenkan, kijun, senkou_a, senkou_b, chikou

def keltner_channels(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20, atr_mult: float = 1.5):
    middle = ema(close, period)
    k_atr = atr(high, low, close, period)
    upper = middle + k_atr * atr_mult
    lower = middle - k_atr * atr_mult
    return upper, middle, lower

def donchian_channel(high: pd.Series, low: pd.Series, period: int = 20):
    upper = high.rolling(period).max()
    lower = low.rolling(period).min()
    middle = (upper + lower) / 2
    return upper, middle, lower

def volume_profile(volume: pd.Series, close: pd.Series, bins: int = 10, period: int = 24):
    result = pd.Series(0.0, index=close.index)
    for i in range(period, len(close)):
        window_vol = volume.iloc[i-period:i]
        window_close = close.iloc[i-period:i]
        if window_vol.sum() == 0:
            continue
        price_min = window_close.min()
        price_max = window_close.max()
        if price_max == price_min:
            continue
        current_price = close.iloc[i]
        bin_idx = min(int((current_price - price_min) / (price_max - price_min) * bins), bins - 1)
        bin_volumes = np.zeros(bins)
        for j in range(period):
            p_idx = min(int((window_close.iloc[j] - price_min) / (price_max - price_min) * bins), bins - 1)
            bin_volumes[p_idx] += window_vol.iloc[j]
        total_vol = bin_volumes.sum()
        if total_vol > 0:
            poc_bin = np.argmax(bin_volumes)
            distance = abs(bin_idx - poc_bin) / bins
            result.iloc[i] = 1 - distance
    return result

def compute_all_features(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or len(df) < 200:
        return df

    df = df.copy()
    o = df['open']
    h = df['high']
    l = df['low']
    c = df['close']
    v = df['volume']

    df['sma_10'] = sma(c, 10)
    df['sma_20'] = sma(c, 20)
    df['sma_50'] = sma(c, 50)
    df['sma_100'] = sma(c, 100)
    df['sma_200'] = sma(c, 200)
    df['ema_9'] = ema(c, 9)
    df['ema_21'] = ema(c, 21)
    df['ema_50'] = ema(c, 50)
    df['ema_200'] = ema(c, 200)

    df['rsi_14'] = rsi(c, 14)

    df['atr_14'] = atr(h, l, c, 14)

    bb_up, bb_mid, bb_low = bollinger_bands(c, 20, 2.0)
    df['bb_upper'] = bb_up
    df['bb_middle'] = bb_mid
    df['bb_lower'] = bb_low
    df['bb_width'] = (bb_up - bb_low) / bb_mid
    df['bb_position'] = (c - bb_low) / (bb_up - bb_low + 1e-10)

    macd_line, signal_line, hist = macd(c, 12, 26, 9)
    df['macd'] = macd_line
    df['macd_signal'] = signal_line
    df['macd_hist'] = hist
    df['macd_hist_roc'] = hist.diff()

    df['adx'], df['plus_di'], df['minus_di'] = adx(h, l, c, 14)

    stoch_k, stoch_d = stochastic(h, l, c, 14, 3)
    df['stoch_k'] = stoch_k
    df['stoch_d'] = stoch_d

    df['mfi'] = money_flow_index(h, l, c, v, 14)

    df['volume_sma_24'] = v.rolling(24).mean()
    df['volume_ratio'] = v / df['volume_sma_24'].replace(0, np.nan)

    df['roc_5'] = c.pct_change(5)
    df['roc_10'] = c.pct_change(10)

    df['ema_slope_9'] = (df['ema_9'] - df['ema_9'].shift(4)) / df['ema_9'].shift(4)
    df['ema_slope_50'] = (df['ema_50'] - df['ema_50'].shift(4)) / df['ema_50'].shift(4)

    df['highest_20'] = h.rolling(20).max()
    df['lowest_20'] = l.rolling(20).min()
    df['highest_50'] = h.rolling(50).max()
    df['lowest_50'] = l.rolling(50).min()

    df['atr_pct'] = df['atr_14'] / c * 100

    return df
