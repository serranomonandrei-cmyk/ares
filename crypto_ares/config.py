from dataclasses import dataclass, field
from typing import List

IN_SAMPLE_START = '2023-01-01'
IN_SAMPLE_END = '2024-12-31'
OUT_OF_SAMPLE_START = '2025-01-01'

TIMEFRAMES = ['15m', '30m', '1h', '4h']

TAKER_FEE = 0.0004
MAKER_FEE = 0.0002
SLIPPAGE = 0.0005

INITIAL_CAPITAL = 50.0
LEVERAGE_MIN = 1
LEVERAGE_MAX = 15
LEVERAGE_DEFAULT = 10

RISK_PER_TRADE = 0.025
MAX_POSITIONS = 5
MAX_DRAWDOWN_SOFT = 0.20
MAX_DRAWDOWN_HARD = 0.35

# Sweet-spot signal params (from sweep)
ENGULF_BODY = 0.7
SWING_DIST = 6
INSIDE_VOL = 1.3
SIGNAL_MIN = 0.35

MIN_VOLUME_USDT = 50_000_000
TOP_N_COINS = 10

DATA_CACHE_DIR = 'data_cache'

@dataclass
class StrategyWeights:
    trend: float = 0.25
    mean_reversion: float = 0.25
    breakout: float = 0.25
    pullback: float = 0.25

REGIME_WEIGHT_MAPS = {
    'trending_bull': StrategyWeights(0.40, 0.10, 0.25, 0.25),
    'trending_bear': StrategyWeights(0.40, 0.10, 0.25, 0.25),
    'ranging_low_vol': StrategyWeights(0.10, 0.45, 0.20, 0.25),
    'ranging_high_vol': StrategyWeights(0.15, 0.20, 0.45, 0.20),
    'strong_trend_bull': StrategyWeights(0.50, 0.05, 0.30, 0.15),
    'strong_trend_bear': StrategyWeights(0.50, 0.05, 0.30, 0.15),
    'default': StrategyWeights(0.25, 0.25, 0.25, 0.25),
}

TRADING_PAIRS = [
    'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT', 'XRP/USDT',
    'ADA/USDT', 'AVAX/USDT', 'DOT/USDT', 'LINK/USDT', 'DOGE/USDT',
    'MATIC/USDT', 'ATOM/USDT', 'UNI/USDT', 'ARB/USDT', 'OP/USDT',
    'APT/USDT', 'LTC/USDT', 'FIL/USDT', 'NEAR/USDT', 'FET/USDT',
]
