import pandas as pd
from binance.client import Client
from pathlib import Path

def download_monthly_price(symbol="BTCUSDT", start_str="1 Oct, 2023", end_str="31 Oct, 2023"):
    print(f"Fetching 1m price data for {symbol} from {start_str} to {end_str}...")

    client = Client()
    klines = client.futures_historical_klines(
        symbol=symbol,
        interval=Client.KLINE_INTERVAL_1MINUTE,
        start_str=start_str,
        end_str=end_str
    )
    
    columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 
               'quote_volume', 'trades', 'taker_base', 'taker_quote', 'ignore']
    
    df_price = pd.DataFrame(klines, columns=columns)
    
    df_price['timestamp'] = pd.to_datetime(df_price['timestamp'], unit='ms')
    df_price['close'] = df_price['close'].astype(float)
    
    df_price = df_price[['timestamp', 'close']]
    df_price.set_index('timestamp', inplace=True)

    out_dir = Path("data/price")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "btc_price_oct_2023.parquet"
    
    df_price.to_parquet(out_path)
    print(f"Success! Saved {len(df_price)} rows of price data to {out_path}")

if __name__ == "__main__":
    download_monthly_price()