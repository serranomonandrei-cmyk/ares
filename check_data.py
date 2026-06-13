import pandas as pd
import os
for sym in ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']:
    safe = sym.replace('/', '_')
    path = f'data_cache/{safe}_15m.parquet'
    if os.path.exists(path):
        df = pd.read_parquet(path)
        print(f'{sym}: {df["timestamp"].min()} to {df["timestamp"].max()} ({len(df)} bars)')
