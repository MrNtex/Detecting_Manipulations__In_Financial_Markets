import pandas as pd
import numpy as np

def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms raw Binance daily depth data into a wide feature matrix 
    ready for unsupervised machine learning.
    """
    
    # We only need the core structural data for now
    df_subset = df[['timestamp', 'percentage', 'depth']]
    
    print("Pivoting data to wide format...")
    # Pivot: index=rows, columns=new features, values=the data to fill in
    matrix_df = df_subset.pivot(index='timestamp', columns='percentage', values='depth')
    
    # Rename columns so they make sense for machine learning
    new_column_names = []
    for pct in matrix_df.columns:
        if pct < 0:
            # Negative percentages are bids (buyers). We use abs() to remove the minus sign
            new_column_names.append(f"bid_depth_{abs(pct)}")
        else:
            # Positive percentages are asks (sellers)
            new_column_names.append(f"ask_depth_{pct}")
            
    matrix_df.columns = new_column_names
    matrix_df = matrix_df.fillna(0)
    
    # --- DYNAMIC FEATURE ENGINEERING ---
    print("Calculating dynamic features...")
    
    # 1. Figure out exactly which percentages are in this specific file
    bid_cols = [col for col in matrix_df.columns if 'bid_depth' in col]
    ask_cols = [col for col in matrix_df.columns if 'ask_depth' in col]
    
    # 2. Sort the actual column names by extracting the float value temporarily for sorting
    # This guarantees we use the exact string Pandas created, avoiding KeyErrors
    bid_cols_sorted = sorted(bid_cols, key=lambda x: float(x.split('_')[2]))
    ask_cols_sorted = sorted(ask_cols, key=lambda x: float(x.split('_')[2]))
    
    # 3. Get the tightest spread (Top of Book)
    top_bid = bid_cols_sorted[0]
    top_ask = ask_cols_sorted[0]
    print(f"Top of book levels found: {top_bid} and {top_ask}")
    
    # Calculate Imbalance at the tightest available spread
    matrix_df['imbalance_top'] = (matrix_df[top_bid] - matrix_df[top_ask]) / \
                                 (matrix_df[top_bid] + matrix_df[top_ask] + 1e-8)
    
    # 4. Get the deepest liquidity (Sum of the 3 deepest levels available)
    deepest_bids = bid_cols_sorted[-3:]
    deepest_asks = ask_cols_sorted[-3:]
    print(f"Deep book levels found: Bids {deepest_bids}, Asks {deepest_asks}")
    
    # Calculate Deep Book Volume
    # sum(axis=1) safely adds multiple columns together row by row
    matrix_df['deep_bid_vol'] = matrix_df[deepest_bids].sum(axis=1)
    matrix_df['deep_ask_vol'] = matrix_df[deepest_asks].sum(axis=1)
    
    print("Transformation complete!")
    return matrix_df

# Run it
if __name__ == "__main__":
    file_name = "YOUR_DOWNLOADED_FILE.zip" 
    
    # Build the matrix
    X_matrix = build_feature_matrix(file_name)
    
    # Inspect the final ML-ready structure
    print("\n=== The Final Feature Matrix (X) ===")
    print(X_matrix.head())
    print(f"\nMatrix Shape: {X_matrix.shape} (Rows: Timestamps, Columns: Features)")