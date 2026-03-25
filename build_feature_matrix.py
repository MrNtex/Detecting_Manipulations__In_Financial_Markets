import pandas as pd
import numpy as np

def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    df_subset = df[['timestamp', 'percentage', 'depth']]
    matrix_df = df_subset.pivot(index='timestamp', columns='percentage', values='depth')
    
    new_column_names = []
    for pct in matrix_df.columns:
        if pct < 0:
            new_column_names.append(f"bid_depth_{abs(pct)}")
        else:
            new_column_names.append(f"ask_depth_{pct}")
            
    matrix_df.columns = new_column_names
    matrix_df = matrix_df.fillna(0)
    
    bid_cols = [col for col in matrix_df.columns if 'bid_depth' in col]
    ask_cols = [col for col in matrix_df.columns if 'ask_depth' in col]
    
    bid_cols_sorted = sorted(bid_cols, key=lambda x: float(x.split('_')[2]))
    ask_cols_sorted = sorted(ask_cols, key=lambda x: float(x.split('_')[2]))
    
    top_bid = bid_cols_sorted[0]
    top_ask = ask_cols_sorted[0]
    print(f"Top of book levels found: {top_bid} and {top_ask}")
    
    matrix_df['imbalance_top'] = (matrix_df[top_bid] - matrix_df[top_ask]) / \
                                 (matrix_df[top_bid] + matrix_df[top_ask] + 1e-8)
    deepest_bids = bid_cols_sorted[-3:]
    deepest_asks = ask_cols_sorted[-3:]
    print(f"Deep book levels found: Bids {deepest_bids}, Asks {deepest_asks}")
    
    matrix_df['deep_bid_vol'] = matrix_df[deepest_bids].sum(axis=1)
    matrix_df['deep_ask_vol'] = matrix_df[deepest_asks].sum(axis=1)
    
    print("Transformation complete!")
    return matrix_df

# Run it
if __name__ == "__main__":
    file_name = "YOUR_DOWNLOADED_FILE.zip" 
    X_matrix = build_feature_matrix(file_name)
    
    print("\n=== The Final Feature Matrix (X) ===")
    print(X_matrix.head())
    print(f"\nMatrix Shape: {X_matrix.shape} (Rows: Timestamps, Columns: Features)")