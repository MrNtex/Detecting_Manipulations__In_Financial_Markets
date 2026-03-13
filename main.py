dir_path = "data/"
file_name = "BTCUSDT-bookDepth-2023-10-16.zip" 

from manage_data import convert_to_dataframe
from detect_anomalies import detect_anomalies
from build_feature_matrix import build_feature_matrix
from visualize_anomaly import visualize_anomaly

if __name__ == "__main__":
    zip_filepath = dir_path + file_name
    raw_df = convert_to_dataframe(zip_filepath)

    X_matrix = build_feature_matrix(zip_filepath)
    
    detect_anomalies(X_matrix)

    X_matrix.index = X_matrix.index.astype(str)
    visualize_anomaly(X_matrix, '2023-10-16 13:00:00')