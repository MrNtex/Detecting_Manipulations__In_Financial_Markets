from pathlib import Path

from sklearn.ensemble import IsolationForest
import pandas as pd

ANOM_DIR = Path("data/anomalies")
ANOM_DIR.mkdir(exist_ok=True)

def detect_anomalies(matrix_df: pd.DataFrame) -> pd.DataFrame:
    iso_forest = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
    
    predictions = iso_forest.fit_predict(matrix_df.values)
    scores = iso_forest.decision_function(matrix_df.values)
    
    results_df = matrix_df.copy()
    results_df['anomaly_label'] = predictions
    results_df['anomaly_score'] = scores
    
    anomalies = results_df[results_df['anomaly_label'] == -1].sort_values(by='anomaly_score')
    
    print(f"\nModel finished! Flagged {len(anomalies)} structural anomalies.")
    
    print("\n=== TOP 5 MOST EXTREME ANOMALIES (The 'Blast Zones') ===")
    cols_to_print = ['anomaly_score', 'imbalance_top', 'deep_bid_vol', 'deep_ask_vol']
    
    print(anomalies[cols_to_print].head(5))
    
    return anomalies

def rolling_detection(feature_files, window=7):

    for i in range(window, len(feature_files)):

        train_files = feature_files[i-window:i]
        test_file = feature_files[i]

        print("\nTraining on:", train_files)
        print("Testing on:", test_file)

        # Load training data
        train_df = pd.concat(
            [pd.read_parquet(f) for f in train_files]
        )

        model = IsolationForest(
            n_estimators=100,
            contamination=0.01,
            random_state=42
        )

        model.fit(train_df.values)

        # Load test day
        test_df = pd.read_parquet(test_file)

        preds = model.predict(test_df.values)
        scores = model.decision_function(test_df.values)

        test_df["anomaly_label"] = preds
        test_df["anomaly_score"] = scores

        anomalies = test_df[test_df["anomaly_label"] == -1]

        date = test_file.name.split("_")[1].replace(".parquet", "")

        anomalies.to_parquet(ANOM_DIR / f"anomalies_{date}.parquet")
        print("Anomalies detected:", len(anomalies))

if __name__ == "__main__":
    rolling_detection(sorted(Path("data/features/").glob("*.parquet")))