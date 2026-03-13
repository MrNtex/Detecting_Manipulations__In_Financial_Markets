dir_path = "data/"
file_name = "BTCUSDT-bookDepth-2026-01-31.zip" 

from manage_data import convert_to_dataframe
from build_feature_matrix import build_feature_matrix

if __name__ == "__main__":
    zip_filepath = dir_path + file_name
    raw_df = convert_to_dataframe(zip_filepath)

    X_matrix = build_feature_matrix(zip_filepath)
    
    print("\n=== The Final Feature Matrix (X) ===")
    print(X_matrix.head())
    print(f"\nMatrix Shape: {X_matrix.shape} (Rows: Timestamps, Columns: Features)")