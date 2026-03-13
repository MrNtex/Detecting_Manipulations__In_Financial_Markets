from sklearn.ensemble import IsolationForest
import pandas as pd

def detect_anomalies(matrix_df: pd.DataFrame) -> pd.DataFrame:
    print("Training Isolation Forest Baseline...")
    
    # Initialize the unsupervised model
    # random_state ensures you get the exact same results every time you run it
    iso_forest = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
    
    # Fit the model and predict outliers
    # IsolationForest returns -1 for anomalies and 1 for normal data points
    predictions = iso_forest.fit_predict(matrix_df.values)
    
    # Calculate the raw mathematical anomaly score 
    # (The more negative the score, the more extreme the anomaly)
    scores = iso_forest.decision_function(matrix_df.values)
    
    # Add the results back to our dataframe so we can read them
    results_df = matrix_df.copy()
    results_df['anomaly_label'] = predictions
    results_df['anomaly_score'] = scores
    
    # Filter out the boring data and sort to find the absolute worst anomalies
    anomalies = results_df[results_df['anomaly_label'] == -1].sort_values(by='anomaly_score')
    
    print(f"\nModel finished! Flagged {len(anomalies)} structural anomalies.")
    
    print("\n=== TOP 5 MOST EXTREME ANOMALIES (The 'Blast Zones') ===")
    # We only print the score and our engineered features to see why the model freaked out
    cols_to_print = ['anomaly_score', 'imbalance_0.2', 'deep_bid_vol', 'deep_ask_vol']
    print(anomalies[cols_to_print].head(5))
    
    return anomalies