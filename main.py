from manage_data import convert_to_dataframe, download_range
from detect_anomalies import detect_anomalies
from build_feature_matrix import build_feature_matrix
from visualize_anomaly import visualize_anomaly

import pandas as pd
from pathlib import Path



if __name__ == "__main__":
    files = download_range("2023-10-01", "2023-10-30")
    for file in files:
        process_day(file)