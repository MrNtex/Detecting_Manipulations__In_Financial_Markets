import pandas as pd
import requests
from pathlib import Path
from datetime import datetime, timedelta
import re

from build_feature_matrix import build_feature_matrix

BASE_URL = "https://data.binance.vision/data/futures/um/daily/bookDepth/BTCUSDT/"
DATA_DIR = Path("data/raw/")
    
def process_day(zip_file):
    raw_df = convert_to_dataframe(zip_file)
    features = build_feature_matrix(raw_df)
    date = re.search(r'\d{4}-\d{2}-\d{2}', zip_file.name).group()
    output = Path(f"data/features/features_{date}.parquet")
    output.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(output)
    
def download_range(start_date: str, end_date: str):

    DATA_DIR.mkdir(exist_ok=True)

    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    current = start
    files = []

    while current <= end:
        date_str = current.strftime("%Y-%m-%d")
        file_name = f"BTCUSDT-bookDepth-{date_str}.zip"
        url = BASE_URL + file_name
        file_path = DATA_DIR / file_name

        if not file_path.exists():
            print("Downloading", file_name)

            r = requests.get(url, stream=True)

            if r.status_code == 200:
                with open(file_path, "wb") as f:
                    for chunk in r.iter_content(8192):
                        f.write(chunk)
            else:
                print("File not found:", file_name)

        files.append(file_path)

        current += timedelta(days=1)

    return files

def inspect_book_depth(zip_filepath: str):
    """
    Reads a zipped Binance historical data file and prints the first 10 rows.
    """
    print(f"Loading data from: {zip_filepath}...")
    
    try:
        df = pd.read_csv(zip_filepath, compression='zip', nrows=10)
        
        print("\n=== First 10 Rows of the Order Book Data ===")
        print(df.to_string())
        
        print("\n=== Column Names ===")
        print(df.columns.tolist())
        
    except Exception as e:
        print(f"Error reading the file: {e}")

def convert_to_dataframe(zip_filepath: str) -> pd.DataFrame:
    
    try:
        df = pd.read_csv(zip_filepath, compression='zip')
        print(f"Data loaded successfully! Total rows: {len(df)}")
        return df
        
    except Exception as e:
        print(f"Error reading the file: {e}")
        return pd.DataFrame()
    

if __name__ == "__main__":
    #files = download_range("2023-10-01", "2023-10-30")
    for file in DATA_DIR.glob("*.zip"):
        process_day(file)