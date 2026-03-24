import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler

from model.model import LSTMAutoencoder
from model.dataset import LOBDataset
from torch.utils.data import DataLoader
from visualize_anomaly import visualize_price_anomalies

FEATURE_DIR = Path("data/features")
ANOM_DIR = Path("data/anomalies")
MODEL_PATH = Path("models/lstm_autoencoder.pth")
TARGET_DAY = "2023-10-16"
SEQ_LEN = 10
BATCH_SIZE = 256

def get_fitted_scaler(first_day="2023-10-01", last_day="2023-10-07"):
    train_dfs = []
    day = first_day
    while day <= last_day:
        file_path = FEATURE_DIR / f"features_{first_day}.parquet"
        if file_path.exists():
            train_dfs.append(pd.read_parquet(file_path))
        day = pd.to_datetime(day) + pd.Timedelta(days=1)
        day = day.strftime("%Y-%m-%d")

    df_train = pd.concat(train_dfs)
    scaler = StandardScaler()
    scaler.fit(df_train.values)
    return scaler, df_train.shape[1]

def run_lstm_inference():
    ANOM_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    scaler, n_features = get_fitted_scaler()
    
    print(f"\nLoading target data for {TARGET_DAY}...")
    df_target = pd.read_parquet(FEATURE_DIR / f"features_{TARGET_DAY}.parquet")
    
    model = LSTMAutoencoder(seq_len=SEQ_LEN, n_features=n_features).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()

    dataset = LOBDataset(df_target, seq_len=SEQ_LEN, scaler=scaler, is_train=False)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    criterion = nn.MSELoss(reduction='none') # 'none' so we get the error for EVERY row
    
    print("Scoring sequences with LSTM...")
    anomaly_scores = []
    
    with torch.no_grad():
        for batch_features, batch_targets in loader:
            batch_features = batch_features.to(device)
            batch_targets = batch_targets.to(device)
            
            reconstruction = model(batch_features)

            loss = criterion(reconstruction, batch_targets)
            seq_loss = loss.mean(dim=(1, 2)).cpu().numpy()
            
            anomaly_scores.extend(seq_loss)

    valid_timestamps = df_target.index[SEQ_LEN:]
    
    df_results = pd.DataFrame(index=valid_timestamps)
    df_results['anomaly_score'] = anomaly_scores
    
    # Invert the logic to match Isolation Forest (lower score = more anomalous)
    df_results['anomaly_score'] = -df_results['anomaly_score'] 
    
    output_path = ANOM_DIR / f"lstm_anomalies_{TARGET_DAY}.parquet"
    df_results.to_parquet(output_path)
    print(f"Saved LSTM anomaly scores to {output_path}")
    visualize_price_anomalies(output_path, TARGET_DAY)

if __name__ == "__main__":
    run_lstm_inference()