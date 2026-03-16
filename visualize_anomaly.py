import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

PLOT_DIR = Path("data/plots")
PLOT_DIR.mkdir(parents=True, exist_ok=True)
PRICE_FILE = Path("data/price/btc_price_oct_2023.parquet")

def visualize_price_anomalies(anomaly_file: Path, date_str: str):
    # 1. Load local anomalies
    try:
        df_anom = pd.read_parquet(anomaly_file)
    except Exception as e:
        print(f"Error reading {anomaly_file}: {e}")
        return

    if df_anom.empty:
        return

    threshold = df_anom['anomaly_score'].quantile(0.03)
    df_anom = df_anom[df_anom['anomaly_score'] <= threshold]
    
    print(f"Plotting {len(df_anom)} anomalies for {date_str}...")
    
    # 2. Load the LOCAL price data (Instant)
    df_price_full = pd.read_parquet(PRICE_FILE)
    
    # Filter the month of price data down to just this specific day
    df_price_day = df_price_full.loc[date_str]

    # 3. Align Anomaly Timestamps with Price Minutes
    anomaly_dt = pd.to_datetime(df_anom.index).floor('min')
    anomaly_prices = df_price_day.loc[df_price_day.index.intersection(anomaly_dt), 'close']

    # 4. Build the Visualization
    fig, ax = plt.subplots(figsize=(14, 7))
    
    ax.plot(df_price_day.index, df_price_day['close'], color='black', linewidth=1.2, label='BTC/USDT Mid-Price')
    ax.scatter(anomaly_prices.index, anomaly_prices.values, color='red', s=100, zorder=5, 
               edgecolor='black', alpha=0.8, label='LOB Structural Anomaly')

    ax.set_title(f"Unsupervised LOB Anomaly Detection vs Price Action ({date_str})", fontsize=14, fontweight='bold')
    ax.set_ylabel("BTC Price (USDT)", fontsize=12)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.xticks(rotation=45)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='upper left')
    
    plt.tight_layout()
    output_path = PLOT_DIR / f"price_anomalies_{date_str}.png"
    plt.savefig(output_path, dpi=150)
    plt.close()