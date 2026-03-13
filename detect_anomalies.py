from sklearn.ensemble import IsolationForest
import pandas as pd

def detect_anomalies(matrix_df: pd.DataFrame) -> pd.DataFrame:
    print("Training Isolation Forest Baseline...")
    
    # Initialize the unsupervised model
    iso_forest = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
    
    # Fit the model and predict outliers
    predictions = iso_forest.fit_predict(matrix_df.values)
    scores = iso_forest.decision_function(matrix_df.values)
    
    results_df = matrix_df.copy()
    results_df['anomaly_label'] = predictions
    results_df['anomaly_score'] = scores
    
    anomalies = results_df[results_df['anomaly_label'] == -1].sort_values(by='anomaly_score')
    
    print(f"\nModel finished! Flagged {len(anomalies)} structural anomalies.")
    
    print("\n=== TOP 5 MOST EXTREME ANOMALIES (The 'Blast Zones') ===")
    
    # FIXED: Changed 'imbalance_0.2' to 'imbalance_top' to match our dynamic naming
    cols_to_print = ['anomaly_score', 'imbalance_top', 'deep_bid_vol', 'deep_ask_vol']
    
    print(anomalies[cols_to_print].head(5))
    
    return anomalies