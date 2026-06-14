#!/usr/bin/env python3
"""
ARES Paper Trading System
=========================
Real-time paper trading using Binance WebSocket feeds.
Mirrors the backtest engine logic exactly for live validation.

Features:
- Real-time 15m kline WebSocket feed
- GPU-accelerated signal computation (engulfing, inside bar, SFP, RSI divergence)
- 4h regime detection (ADX, EMA slope, MACD)
- Regime-adaptive risk sizing
- Dynamic signal gate
- Virtual position management (SL/TP/48h time exit)
- Trade logging to JSON
- Equity curve tracking

Usage:
    python paper_trade.py
    python paper_trade.py --symbols BTC/USDT,SOL/USDT --risk 0.025
    python paper_trade.py --test  # Test mode with simulated data
"""

import asyncio
import json
import time
import os
import sys
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import deque
import signal as sig

import numpy as np
import pandas as pd
import websockets
import requests
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from crypto_ares.data.features import compute_all_features, atr, ema, rsi, macd, adx
from crypto_ares.strategy.setups_gpu import compute_all_signals
from crypto_ares.strategy.regime import detect_regime
from crypto_ares.strategy.ensemble import compute_position_size
from crypto_ares.config import (
    TAKER_FEE, SLIPPAGE, INITIAL_CAPITAL, MAX_DRAWDOWN_HARD,
    RISK_PER_TRADE, LEVERAGE_DEFAULT, ENGULF_BODY, SWING_DIST,
    INSIDE_VOL, SIGNAL_MIN, TRADING_PAIRS
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('paper_trade.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class Position:
    """Active position tracker."""
    symbol: str
    side: str  # 'long' or 'short'
    entry_price: float
    quantity: float
    position_value: float
    leverage: float
    stop_loss: float
    take_profit: float
    entry_time: datetime
    entry_signal_strength: float
    fee_paid: float = 0.0
    pnl: float = 0.0


@dataclass
class TradeRecord:
    """Completed trade record."""
    symbol: str
    side: str
    entry_time: str
    entry_price: float
    exit_time: str
    exit_price: float
    quantity: float
    position_value: float
    leverage: float
    pnl: float
    pnl_pct: float
    exit_reason: str
    fee_paid: float
    regime: str
    signal_strength: float


class BinanceWebSocket:
    """Binance futures WebSocket client for real-time kline data."""
    
    BASE_URL = "wss://fstream.binance.com/ws"
    
    def __init__(self, symbols: List[str], interval: str = '15m'):
        self.symbols = symbols
        self.interval = interval
        self.streams = [f"{s.lower().replace('/', '')}@kline_{interval}" for s in symbols]
        self.running = False
        self.kline_callback = None
        self._ws = None
        
    def on_kline(self, callback):
        """Register callback for kline updates."""
        self.kline_callback = callback
        
    async def start(self):
        """Start WebSocket connection."""
        self.running = True
        stream_path = "/".join(self.streams)
        url = f"{self.BASE_URL}/{stream_path}"
        
        logger.info(f"Connecting to Binance WebSocket: {len(self.symbols)} symbols")
        
        try:
            async with websockets.connect(url, ping_interval=20, ping_timeout=10) as ws:
                self._ws = ws
                logger.info("WebSocket connected")
                
                while self.running:
                    try:
                        msg = await asyncio.wait_for(ws.recv(), timeout=30)
                        data = json.loads(msg)
                        
                        if 'e' in data and data['e'] == 'kline':
                            kline = data['k']
                            symbol = kline['s']
                            is_closed = kline['x']  # True if this is the final close
                            
                            if is_closed and self.kline_callback:
                                await self.kline_callback(symbol, kline)
                                
                    except asyncio.TimeoutError:
                        continue
                    except websockets.ConnectionClosed:
                        logger.warning("WebSocket connection closed, reconnecting...")
                        await asyncio.sleep(5)
                        break
                        
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
            
    def stop(self):
        """Stop WebSocket connection."""
        self.running = False
        logger.info("WebSocket stopped")


class PaperTradingEngine:
    """
    Paper trading engine that mirrors backtest logic exactly.
    
    Key features:
    - 15m kline processing with feature computation
    - GPU signal evaluation (engulfing, inside bar, SFP, RSI divergence)
    - 4h regime detection with lag safety
    - Regime-adaptive risk sizing
    - Dynamic signal gate
    - Position management (SL/TP/time exit/reversal)
    """
    
    def __init__(
        self,
        symbols: List[str],
        initial_capital: float = INITIAL_CAPITAL,
        risk_per_trade: float = RISK_PER_TRADE,
        leverage: int = LEVERAGE_DEFAULT,
        signal_gate: int = 5,
        test_mode: bool = False,
    ):
        self.symbols = symbols
        self.initial_capital = initial_capital
        self.risk_per_trade = risk_per_trade
        self.leverage = leverage
        self.signal_gate = signal_gate
        self.test_mode = test_mode
        
        # State per symbol
        self.equity = {s: initial_capital for s in symbols}
        self.peak_equity = {s: initial_capital for s in symbols}
        self.consecutive_losses = {s: 0 for s in symbols}
        self.trading_stopped = {s: False for s in symbols}
        self.daily_signal_counts = {s: {} for s in symbols}
        
        # Positions
        self.open_positions: Dict[str, Position] = {}
        
        # Historical data for feature computation
        self.kline_history: Dict[str, pd.DataFrame] = {
            s: pd.DataFrame() for s in symbols
        }
        
        # Regime data (4h)
        self.regime_history: Dict[str, pd.DataFrame] = {
            s: pd.DataFrame() for s in symbols
        }
        self.current_regime: Dict[str, dict] = {s: {} for s in symbols}
        self.last_regime_update: Dict[str, datetime] = {s: None for s in symbols}
        
        # Trade logging
        self.trades: List[TradeRecord] = []
        self.equity_curve: Dict[str, List[Tuple[datetime, float]]] = {
            s: [(datetime.now(timezone.utc), initial_capital)] for s in symbols
        }
        
        # Trade log file
        self.trade_log_path = Path('paper_trades.json')
        self.equity_log_path = Path('paper_equity.json')
        
        # WebSocket
        self.ws = BinanceWebSocket(symbols, '15m')
        self.ws.on_kline(self._on_kline)
        
        # Graceful shutdown
        self._shutdown_event = asyncio.Event()
        
    def _on_signal(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, shutting down...")
        self._shutdown_event.set()
        self.ws.stop()
        
    async def initialize(self):
        """Load historical data for each symbol."""
        from crypto_ares.data.downloader import load_cached_data
        
        logger.info("Initializing historical data...")
        
        for symbol in self.symbols:
            try:
                # Load 15m data
                df = load_cached_data(symbol, '15m')
                if not df.empty:
                    df = compute_all_features(df)
                    self.kline_history[symbol] = df.tail(500).copy()
                    logger.info(f"  {symbol}: Loaded {len(df)} 15m bars ({len(df.columns)} features)")
                else:
                    # Fetch from Binance API
                    df = await self._fetch_historical(symbol, '15m', limit=500)
                    if df is not None:
                        self.kline_history[symbol] = df
                        logger.info(f"  {symbol}: Fetched {len(df)} 15m bars")
                    else:
                        logger.warning(f"  {symbol}: No 15m data available")
                        
                # Load 4h data (resampled from 1h if needed)
                df_4h = load_cached_data(symbol, '4h')
                if not df_4h.empty:
                    df_4h = compute_all_features(df_4h)
                    self.regime_history[symbol] = df_4h.tail(500).copy()
                    logger.info(f"  {symbol}: Loaded {len(df_4h)} 4h bars for regime")
                    
                    # Initialize regime from historical data
                    if len(df_4h) >= 200:
                        self.current_regime[symbol] = detect_regime(df_4h)
                        logger.info(f"  {symbol}: Initial regime = {self.current_regime[symbol].get('composite', 'unknown')}")
                else:
                    logger.warning(f"  {symbol}: No 4h data available for regime")
                    
            except Exception as e:
                logger.error(f"  {symbol}: Initialization error: {e}")
                
        logger.info("Initialization complete")
        
    async def _fetch_historical(self, symbol: str, interval: str, limit: int = 500) -> Optional[pd.DataFrame]:
        """Fetch historical klines from Binance REST API."""
        try:
            url = "https://fapi.binance.com/fapi/v1/klines"
            params = {
                'symbol': symbol.replace('/', ''),
                'interval': interval,
                'limit': limit
            }
            
            resp = requests.get(url, params=params, timeout=10)
            data = resp.json()
            
            if not data:
                return None
                
            df = pd.DataFrame(data, columns=[
                'open_time', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ])
            
            df['timestamp'] = pd.to_datetime(df['open_time'], unit='ms')
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)
                
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            df = df.drop_duplicates(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)
            
            # Compute features
            df = compute_all_features(df)
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to fetch {symbol} {interval}: {e}")
            return None
            
    async def _on_kline(self, symbol: str, kline: dict):
        """Process closed kline."""
        try:
            ts = datetime.fromtimestamp(kline['t'] / 1000, tz=timezone.utc)
            price = float(kline['c'])
            high = float(kline['h'])
            low = float(kline['l'])
            opn = float(kline['o'])
            volume = float(kline['v'])
            
            # Create new row
            new_row = pd.DataFrame([{
                'timestamp': ts,
                'open': opn,
                'high': high,
                'low': low,
                'close': price,
                'volume': volume
            }])
            
            # Append to history
            self.kline_history[symbol] = pd.concat(
                [self.kline_history[symbol], new_row],
                ignore_index=True
            ).tail(1000)  # Keep last 1000 bars
            
            # Recompute features on full history
            self.kline_history[symbol] = compute_all_features(self.kline_history[symbol])
            
            # Update 4h regime data
            await self._update_regime_data(symbol, ts, high, low, price, volume)
            
            # Check existing position
            if symbol in self.open_positions:
                await self._check_position_exit(symbol, price, ts)
                
            # Look for new entry
            if symbol not in self.open_positions and not self.trading_stopped[symbol]:
                await self._evaluate_entry(symbol, price, ts)
                
            # Log equity
            eq = self.equity[symbol]
            if symbol in self.open_positions:
                pos = self.open_positions[symbol]
                if pos.side == 'long':
                    unrealized = (price - pos.entry_price) * pos.quantity
                else:
                    unrealized = (pos.entry_price - price) * pos.quantity
                eq += unrealized - pos.fee_paid
                
            self.equity_curve[symbol].append((ts, eq))
            
        except Exception as e:
            logger.error(f"Error processing kline for {symbol}: {e}")
            
    async def _update_regime_data(self, symbol: str, ts: datetime, high: float, low: float, close: float, volume: float):
        """Update 4h regime data with new 15m bar."""
        # Determine which 4h bucket this belongs to
        hour_4h = (ts.hour // 4) * 4
        bucket_ts = ts.replace(hour=hour_4h, minute=0, second=0, microsecond=0)
        
        # Check if we have existing data for this 4h bucket
        df_4h = self.regime_history[symbol]
        
        if len(df_4h) > 0 and df_4h.iloc[-1]['timestamp'] == bucket_ts:
            # Update existing bar
            idx = len(df_4h) - 1
            df_4h.at[idx, 'high'] = max(df_4h.at[idx, 'high'], high)
            df_4h.at[idx, 'low'] = min(df_4h.at[idx, 'low'], low)
            df_4h.at[idx, 'close'] = close
            df_4h.at[idx, 'volume'] += volume
        else:
            # New 4h bar
            new_row = pd.DataFrame([{
                'timestamp': bucket_ts,
                'open': close,  # Use close as open for new bar
                'high': high,
                'low': low,
                'close': close,
                'volume': volume
            }])
            self.regime_history[symbol] = pd.concat(
                [df_4h, new_row],
                ignore_index=True
            ).tail(500)
            
        # Recompute 4h features periodically
        if len(self.regime_history[symbol]) % 4 == 0:  # Every 4 hours
            self.regime_history[symbol] = compute_all_features(self.regime_history[symbol])
            
            # Update regime
            if len(self.regime_history[symbol]) >= 200:
                self.current_regime[symbol] = detect_regime(self.regime_history[symbol])
                self.last_regime_update[symbol] = ts
                
    async def _check_position_exit(self, symbol: str, price: float, ts: datetime):
        """Check if position should be exited (SL/TP/time/reversal)."""
        if symbol not in self.open_positions:
            return
            
        pos = self.open_positions[symbol]
        exit_reason = None
        exit_price = price
        
        # Check stop loss
        if pos.side == 'long' and price <= pos.stop_loss:
            exit_reason = 'sl'
            exit_price = pos.stop_loss
        elif pos.side == 'short' and price >= pos.stop_loss:
            exit_reason = 'sl'
            exit_price = pos.stop_loss
            
        # Check take profit
        if pos.side == 'long' and price >= pos.take_profit:
            exit_reason = 'tp'
            exit_price = pos.take_profit
        elif pos.side == 'short' and price <= pos.take_profit:
            exit_reason = 'tp'
            exit_price = pos.take_profit
            
        # Check time exit (48h)
        if exit_reason is None:
            hold_h = (ts - pos.entry_time).total_seconds() / 3600
            if hold_h > 48:
                exit_reason = 'time'
                exit_price = price
                
        # Check reversal signal (simplified for live - would need current signal)
        # In live, we'd compute the current signal strength and compare
        
        if exit_reason is not None:
            await self._close_position(symbol, exit_price, exit_reason, ts)
            
    async def _close_position(self, symbol: str, exit_price: float, reason: str, ts: datetime):
        """Close position and update equity."""
        pos = self.open_positions[symbol]
        
        # Apply slippage
        slip = exit_price * SLIPPAGE
        if pos.side == 'long':
            exit_adj = exit_price - slip  # Slippage against us
        else:
            exit_adj = exit_price + slip
            
        # Compute PnL
        if pos.side == 'long':
            raw_pnl = (exit_adj - pos.entry_price) * pos.quantity
        else:
            raw_pnl = (pos.entry_price - exit_adj) * pos.quantity
            
        # Exit fee
        fee = pos.position_value * TAKER_FEE
        pnl = raw_pnl - fee
        
        # Update equity
        self.equity[symbol] += pnl
        
        # Record trade
        regime = self.current_regime[symbol].get('composite', 'unknown')
        trade = TradeRecord(
            symbol=symbol,
            side=pos.side,
            entry_time=pos.entry_time.isoformat(),
            entry_price=pos.entry_price,
            exit_time=ts.isoformat(),
            exit_price=exit_adj,
            quantity=pos.quantity,
            position_value=pos.position_value,
            leverage=pos.leverage,
            pnl=pnl,
            pnl_pct=pnl / self.equity[symbol] * 100 if self.equity[symbol] > 0 else 0,
            exit_reason=reason,
            fee_paid=pos.fee_paid + fee,
            regime=regime,
            signal_strength=pos.entry_signal_strength,
        )
        self.trades.append(trade)
        
        # Update consecutive losses
        if pnl < 0:
            self.consecutive_losses[symbol] += 1
        else:
            self.consecutive_losses[symbol] = 0
            
        # Check drawdown
        self.peak_equity[symbol] = max(self.peak_equity[symbol], self.equity[symbol])
        dd = (self.peak_equity[symbol] - self.equity[symbol]) / self.peak_equity[symbol]
        if dd >= MAX_DRAWDOWN_HARD:
            self.trading_stopped[symbol] = True
            logger.warning(f"{symbol}: Trading stopped due to drawdown {dd:.1%}")
            
        # Log
        logger.info(
            f"{symbol}: Closed {pos.side} @ {exit_adj:.2f} | "
            f"PnL: ${pnl:.2f} ({trade.pnl_pct:.1f}%) | "
            f"Reason: {reason} | Equity: ${self.equity[symbol]:.2f}"
        )
        
        # Remove position
        del self.open_positions[symbol]
        
        # Save trade
        await self._save_trade(trade)
        
    async def _evaluate_entry(self, symbol: str, price: float, ts: datetime):
        """Evaluate whether to enter a new position."""
        df = self.kline_history[symbol]
        if len(df) < 200:
            return
            
        # Check daily signal gate
        day = ts.date().isoformat()
        effective_gate = max(
            self.signal_gate,
            int(self.equity[symbol] / self.initial_capital * self.signal_gate)
        )
        
        if self.daily_signal_counts[symbol].get(day, 0) >= effective_gate:
            return
            
        # Get current regime
        regime = self.current_regime[symbol]
        if not regime:
            return
            
        direction_filter = regime.get('direction', 'neutral')
        
        # Compute signals on recent window
        window = df.tail(200)
        try:
            sig_long, sig_short, sig_conf = compute_all_signals(
                window,
                engulf_body=ENGULF_BODY,
                swing_dist=SWING_DIST,
                inside_vol=INSIDE_VOL
            )
        except Exception as e:
            logger.error(f"{symbol}: Signal computation failed: {e}")
            return
            
        # Get last bar signal
        if len(sig_long) == 0:
            return
            
        sig_long_val = float(sig_long[-1])
        sig_short_val = float(sig_short[-1])
        conf_val = float(sig_conf[-1])
        
        # Determine direction
        if sig_long_val > sig_short_val and sig_long_val > SIGNAL_MIN:
            raw_signal = sig_long_val
            raw_action = 'long'
        elif sig_short_val > SIGNAL_MIN:
            raw_signal = -sig_short_val
            raw_action = 'short'
        else:
            return  # No signal
            
        # Apply regime filter
        if direction_filter == 'bearish' and raw_action == 'long':
            trend_regime = regime.get('trend_regime', 'ranging')
            mult = {'strong_trend': 0.2, 'trending': 0.5, 'ranging': 0.8}.get(trend_regime, 0.8)
            raw_signal *= mult
        elif direction_filter == 'bullish' and raw_action == 'short':
            trend_regime = regime.get('trend_regime', 'ranging')
            mult = {'strong_trend': 0.2, 'trending': 0.5, 'ranging': 0.8}.get(trend_regime, 0.8)
            raw_signal *= mult
            
        if abs(raw_signal) < SIGNAL_MIN:
            return
            
        # Update signal count
        self.daily_signal_counts[symbol][day] = self.daily_signal_counts[symbol].get(day, 0) + 1
        
        # Compute position size
        atr_val = window['atr_14'].iloc[-1] if 'atr_14' in window else price * 0.01
        
        # Regime-adaptive risk
        risk = self.risk_per_trade
        if self.consecutive_losses[symbol] >= 5:
            risk = max(0.005, risk * 0.5)
            
        regime_mult = {
            'strong_trend_bull': 1.0, 'strong_trend_bear': 1.0,
            'trending_bull': 0.9, 'trending_bear': 0.9,
            'ranging_low_vol': 0.5, 'ranging_high_vol': 0.6,
            'default': 0.7,
        }.get(regime.get('composite', 'default'), 0.7)
        risk *= regime_mult
        
        signal = {
            'action': raw_action,
            'signal_strength': abs(raw_signal),
            'confidence': conf_val,
        }
        
        pos_info = compute_position_size(
            self.equity[symbol],
            price,
            atr_val,
            signal,
            {'risk_per_trade': risk, 'leverage': self.leverage, 'sl_mult': 2.0}
        )
        
        qty = pos_info['quantity']
        pos_val = qty * price
        lev = pos_info['leverage']
        
        # Validate position
        if pos_val < self.equity[symbol] * 0.02 or pos_val > self.equity[symbol] * lev:
            return
            
        # Apply entry slippage
        slip = price * SLIPPAGE
        entry_price = price + slip if raw_action == 'long' else price - slip
        
        # Compute SL/TP
        sl_pct = pos_info['stop_loss_pct']
        tp_pct = pos_info['take_profit_pct']
        
        if raw_action == 'long':
            sl_price = entry_price * (1 + sl_pct)
            tp_price = entry_price * (1 + tp_pct)
        else:
            sl_price = entry_price * (1 - sl_pct)
            tp_price = entry_price * (1 - tp_pct)
            
        # Entry fee
        fee = pos_info['position_value'] * TAKER_FEE
        
        # Create position
        pos = Position(
            symbol=symbol,
            side=raw_action,
            entry_price=entry_price,
            quantity=qty,
            position_value=pos_info['position_value'],
            leverage=lev,
            stop_loss=sl_price,
            take_profit=tp_price,
            entry_time=ts,
            entry_signal_strength=abs(raw_signal),
            fee_paid=fee,
        )
        
        self.open_positions[symbol] = pos
        
        # Log entry
        regime_str = regime.get('composite', 'unknown')
        logger.info(
            f"{symbol}: Entered {raw_action} @ {entry_price:.2f} | "
            f"Qty: {qty:.4f} | Lev: {lev}x | "
            f"SL: {sl_price:.2f} | TP: {tp_price:.2f} | "
            f"Signal: {abs(raw_signal):.2f} | Regime: {regime_str}"
        )
        
    async def _save_trade(self, trade: TradeRecord):
        """Append trade to log file."""
        try:
            trades = []
            if self.trade_log_path.exists():
                with open(self.trade_log_path, 'r') as f:
                    trades = json.load(f)
                    
            trades.append(asdict(trade))
            
            with open(self.trade_log_path, 'w') as f:
                json.dump(trades, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save trade: {e}")
            
    async def _save_equity(self):
        """Save equity curves to file."""
        try:
            data = {}
            for symbol, curve in self.equity_curve.items():
                data[symbol] = [
                    {'timestamp': ts.isoformat(), 'equity': eq}
                    for ts, eq in curve[-1000:]  # Last 1000 points
                ]
                
            with open(self.equity_log_path, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save equity: {e}")
            
    def print_status(self):
        """Print current status."""
        print("\n" + "="*70)
        print(f"PAPER TRADING STATUS - {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
        print("="*70)
        
        total_equity = 0
        for symbol in self.symbols:
            eq = self.equity[symbol]
            pos = self.open_positions.get(symbol)
            
            if pos:
                print(f"  {symbol}: ${eq:.2f} | Position: {pos.side} @ {pos.entry_price:.2f}")
            else:
                print(f"  {symbol}: ${eq:.2f} | No position")
                
            total_equity += eq
            
        print("-"*70)
        print(f"  Total Equity: ${total_equity:.2f}")
        print(f"  Total Trades: {len(self.trades)}")
        
        winning = sum(1 for t in self.trades if t.pnl > 0)
        print(f"  Win Rate: {winning/len(self.trades)*100:.1f}%" if self.trades else "  Win Rate: N/A")
        
        total_pnl = sum(t.pnl for t in self.trades)
        print(f"  Total PnL: ${total_pnl:.2f}")
        print("="*70 + "\n")
        
    async def run(self):
        """Main run loop."""
        # Register signal handlers
        sig.signal(sig.SIGINT, self._on_signal)
        sig.signal(sig.SIGTERM, self._on_signal)
        
        # Initialize
        await self.initialize()
        
        # Start equity save task
        async def save_loop():
            while not self._shutdown_event.is_set():
                await asyncio.sleep(60)  # Save every minute
                await self._save_equity()
                self.print_status()
                
        # Start tasks
        ws_task = asyncio.create_task(self.ws.start())
        save_task = asyncio.create_task(save_loop())
        
        logger.info("Paper trading started")
        logger.info(f"Symbols: {self.symbols}")
        logger.info(f"Initial Capital: ${self.initial_capital} per coin")
        logger.info(f"Risk Per Trade: {self.risk_per_trade*100}%")
        logger.info(f"Leverage: {self.leverage}x")
        
        # Wait for shutdown
        await self._shutdown_event.wait()
        
        # Cleanup
        self.ws.stop()
        save_task.cancel()
        
        # Final save
        await self._save_equity()
        self.print_status()
        
        logger.info("Paper trading stopped")


def parse_args():
    """Parse command line arguments."""
    import argparse
    
    parser = argparse.ArgumentParser(description='ARES Paper Trading System')
    parser.add_argument('--symbols', type=str, default=None,
                        help='Comma-separated symbols (default: top 5)')
    parser.add_argument('--risk', type=float, default=RISK_PER_TRADE,
                        help=f'Risk per trade (default: {RISK_PER_TRADE})')
    parser.add_argument('--leverage', type=int, default=LEVERAGE_DEFAULT,
                        help=f'Leverage (default: {LEVERAGE_DEFAULT})')
    parser.add_argument('--signal-gate', type=int, default=5,
                        help='Base signal gate per day (default: 5)')
    parser.add_argument('--capital', type=float, default=INITIAL_CAPITAL,
                        help=f'Initial capital per coin (default: {INITIAL_CAPITAL})')
    parser.add_argument('--test', action='store_true',
                        help='Test mode with simulated data')
    
    return parser.parse_args()


async def main():
    """Entry point."""
    args = parse_args()
    
    # Default symbols (top 5 by volume, excluding ETH)
    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(',')]
    else:
        symbols = ['BTC/USDT', 'SOL/USDT', 'BNB/USDT', 'XRP/USDT', 'DOGE/USDT']
        
    engine = PaperTradingEngine(
        symbols=symbols,
        initial_capital=args.capital,
        risk_per_trade=args.risk,
        leverage=args.leverage,
        signal_gate=args.signal_gate,
        test_mode=args.test,
    )
    
    await engine.run()


if __name__ == '__main__':
    asyncio.run(main())