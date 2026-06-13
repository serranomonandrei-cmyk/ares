import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from crypto_ares.data.downloader import download_all_data
from crypto_ares.config import TRADING_PAIRS

symbols = TRADING_PAIRS[:10]
print(f'Downloading {len(symbols)} symbols: {symbols}')
data = download_all_data(symbols=symbols)
for k, v in data.items():
    print(f'  {k}: {len(v)} rows from {v["timestamp"].min()} to {v["timestamp"].max()}')
print('Download complete!')
