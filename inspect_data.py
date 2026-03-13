import pandas as pd

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

dir_path = "data/"
file_name = "BTCUSDT-bookDepth-2026-01-31.zip" 

if __name__ == "__main__":
    inspect_book_depth(dir_path + file_name)