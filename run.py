#!/usr/bin/env python3
"""
ARES - Adaptive Regime Ensemble Strategy
Multi-timeframe futures trading bot for Binance
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from crypto_ares.app.dashboard import run_dashboard

if __name__ == '__main__':
    run_dashboard()
