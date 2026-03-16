from pathlib import Path
import pandas as pd
import re

from manage_data import download_range, process_day
from detect_anomalies import rolling_detection
from visualize_heatmap import visualize_top_heatmaps
from visualize_anomaly import visualize_price_anomalies 

RAW_DIR = Path("data/raw")
FEATURE_DIR = Path("data/features")
ANOM_DIR = Path("data/anomalies")

START = "2023-10-01"
END = "2023-10-30"

def ensure_features():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    FEATURE_DIR.mkdir(parents=True, exist_ok=True)

    raw_files = download_range(START, END)
    feature_files = []

    for raw_file in raw_files:
        date = re.search(r'\d{4}-\d{2}-\d{2}', raw_file.name).group()
        feature_path = FEATURE_DIR / f"features_{date}.parquet"

        if feature_path.exists():
            print("Feature file exists:", feature_path)
        else:
            print("Processing:", raw_file)
            process_day(raw_file)

        feature_files.append(feature_path)

    return sorted(feature_files)

def main():
    feature_files = ensure_features()

    rolling_detection(feature_files, window=7)
    
    for anomaly_file in sorted(ANOM_DIR.glob("*.parquet")):
        date_str = anomaly_file.stem.split('_')[-1] 
        #visualize_top_heatmaps(anomaly_file)
        visualize_price_anomalies(anomaly_file,f"2023-10-{date_str}")

if __name__ == "__main__":
    main()