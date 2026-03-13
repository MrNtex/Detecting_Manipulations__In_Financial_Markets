import pandas as pd
import numpy as np

def build_feature_matrix(zip_filepath: str) -> pd.DataFrame:
    """
    Transforms raw Binance daily depth data into a wide feature matrix 
    ready for unsupervised machine learning.
    """
    df = pd.read_csv(zip_filepath, compression='zip')
    
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
    
    # Fill missing values. If a percentage level has no data at a specific second, depth is 0.
    matrix_df = matrix_df.fillna(0)
    
    # --- FEATURE ENGINEERING ---
    # Now that the data is wide, we can calculate structural anomalies instantly.
    
    # Example: Imbalance at the tightest spread (0.2%)
    # If this violently shifts, someone is spoofing the top of the book.
    # We add 1e-8 to the denominator to prevent division by zero errors.
    matrix_df['imbalance_0.2'] = (matrix_df['bid_depth_0.2'] - matrix_df['ask_depth_0.2']) / \
                                 (matrix_df['bid_depth_0.2'] + matrix_df['ask_depth_0.2'] + 1e-8)
    
    # Example: Deep Book Volume (Sum of 3%, 4%, and 5% levels)
    # Spoofers hide fake walls deep in the book.
    matrix_df['deep_bid_vol'] = matrix_df['bid_depth_3.0'] + matrix_df['bid_depth_4.0'] + matrix_df['bid_depth_5.0']
    matrix_df['deep_ask_vol'] = matrix_df['ask_depth_3.0'] + matrix_df['ask_depth_4.0'] + matrix_df['ask_depth_5.0']
    
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