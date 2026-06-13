import os
import time
import json
from datetime import datetime, timezone
from pathlib import Path

import ccxt
import pandas as pd
import numpy as np

from ..config import DATA_CACHE_DIR, TIMEFRAMES, TRADING_PAIRS, MIN_VOLUME_USDT

CACHE_DIR = Path(DATA_CACHE_DIR)
CACHE_DIR.mkdir(exist_ok=True)

def get_binance():
    exchange = ccxt.binance({
        'enableRateLimit': True,
        'options': {'defaultType': 'future'},
    })
    exchange.load_markets()
    return exchange

def fetch_ohlcv(exchange, symbol: str, timeframe: str, since: int, limit: int = 1000):
    all_candles = []
    while True:
        candles = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
        if not candles:
            break
        all_candles.extend(candles)
        if len(candles) < limit:
            break
        since = candles[-1][0] + 1
        time.sleep(exchange.rateLimit / 1000)
    return all_candles

def cache_path(symbol: str, timeframe: str) -> Path:
    safe_name = symbol.replace('/', '_').replace(' ', '_')
    return CACHE_DIR / f"{safe_name}_{timeframe}.parquet"

def download_symbol(exchange, symbol: str, timeframe: str, start_ts: int, force: bool = False) -> pd.DataFrame:
    cache = cache_path(symbol, timeframe)
    start_dt = pd.Timestamp(start_ts, unit='ms')
    if cache.exists() and not force:
        df = pd.read_parquet(cache)
        df = df[df['timestamp'] >= start_dt]
        if len(df) > 100:
            return df

    pair = symbol.replace('BTC/USDT', 'BTC/USDT:USDT')
    try:
        candles = fetch_ohlcv(exchange, symbol, timeframe, since=start_ts)
    except Exception as e:
        print(f"  Error downloading {symbol} {timeframe}: {e}")
        return pd.DataFrame()

    if not candles:
        return pd.DataFrame()

    df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    df['timestamp'] = df['timestamp'].dt.tz_localize(None)
    df['symbol'] = symbol
    df = df.drop_duplicates(subset='timestamp').sort_values('timestamp')
    df.to_parquet(cache, index=False)
    print(f"  Cached {symbol} {timeframe}: {len(df)} candles")
    return df

def resample_4h(df_1h: pd.DataFrame) -> pd.DataFrame:
    df = df_1h.set_index('timestamp').resample('4h').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
    }).dropna().reset_index()
    if 'symbol' in df_1h.columns:
        df['symbol'] = df_1h['symbol'].iloc[0]
    return df

def resample_30m(df_15m: pd.DataFrame) -> pd.DataFrame:
    df = df_15m.set_index('timestamp').resample('30min').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
    }).dropna().reset_index()
    if 'symbol' in df_15m.columns:
        df['symbol'] = df_15m['symbol'].iloc[0]
    return df

def download_all_data(symbols: list = None, force: bool = False) -> dict:
    if symbols is None:
        symbols = TRADING_PAIRS

    exchange = get_binance()
    now_ts = int(datetime.now(timezone.utc).timestamp() * 1000)
    start_ts = int(datetime(2023, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)

    result = {}
    for symbol in symbols:
        print(f"Downloading {symbol}...")
        df_1h = download_symbol(exchange, symbol, '1h', start_ts, force)
        df_15m = download_symbol(exchange, symbol, '15m', start_ts, force)
        if df_1h.empty or df_15m.empty:
            continue
        df_4h = resample_4h(df_1h)
        df_30m = resample_30m(df_15m)
        result[f"{symbol}_1h"] = df_1h
        result[f"{symbol}_4h"] = df_4h
        result[f"{symbol}_15m"] = df_15m
        result[f"{symbol}_30m"] = df_30m
        print(f"  {symbol}: 1h={len(df_1h)}, 4h={len(df_4h)}, 15m={len(df_15m)}, 30m={len(df_30m)}")

    return result

def get_top_volume_pairs(n: int = 10) -> list:
    exchange = get_binance()
    try:
        tickers = exchange.fetch_tickers()
        usdt_pairs = []
        for sym, t in tickers.items():
            if sym.endswith('/USDT') and ':' not in sym:
                vol = t.get('quoteVolume', 0) or 0
                if vol > MIN_VOLUME_USDT:
                    usdt_pairs.append((sym, vol))
        usdt_pairs.sort(key=lambda x: x[1], reverse=True)
        return [p[0] for p in usdt_pairs[:n]]
    except Exception as e:
        print(f"Error fetching top volume pairs: {e}")
        return TRADING_PAIRS[:n]

def load_cached_data(symbol: str, timeframe: str) -> pd.DataFrame:
    cache = cache_path(symbol, timeframe)
    if cache.exists():
        return pd.read_parquet(cache)
    if timeframe == '4h':
        df_1h = load_cached_data(symbol, '1h')
        if not df_1h.empty:
            return resample_4h(df_1h)
    if timeframe == '30m':
        df_15m = load_cached_data(symbol, '15m')
        if not df_15m.empty:
            return resample_30m(df_15m)
    return pd.DataFrame()

def list_cached_symbols() -> list:
    symbols = set()
    for f in CACHE_DIR.glob("*.parquet"):
        parts = f.stem.split('_')
        if len(parts) >= 2:
            tf = parts[-1]
            sym = '_'.join(parts[:-1]).replace('_', '/')
            symbols.add(sym)
    return sorted(symbols)
