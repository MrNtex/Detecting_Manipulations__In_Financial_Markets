import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from pathlib import Path

from model import LSTMAutoencoder
from dataset import create_dataloaders

FEATURE_DIR = Path("data/features")
SEQ_LEN = 60
BATCH_SIZE = 256
EPOCHS = 15
LEARNING_RATE = 0.001

def load_training_data():
    print("Loading Parquet files for training baseline...")
    train_dfs = []
    
    # Load days 1 through 7
    for day in range(1, 8):
        file_path = FEATURE_DIR / f"features_2023-10-{day:02d}.parquet"
        if file_path.exists():
            train_dfs.append(pd.read_parquet(file_path))
            
    # Combine them into one massive DataFrame
    
    df_train = pd.concat(train_dfs)
    
    # Load Day 8 as our validation set to ensure we aren't overfitting
    df_val = pd.read_parquet(FEATURE_DIR / "features_2023-10-08.parquet")
    
    return df_train, df_val

def train_autoencoder():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    df_train, df_val = load_training_data()
    n_features = df_train.shape[1]
    
    train_loader, val_loader, scaler = create_dataloaders(
        df_train, df_val, seq_len=SEQ_LEN, batch_size=BATCH_SIZE
    )

    model = LSTMAutoencoder(seq_len=SEQ_LEN, n_features=n_features, embedding_dim=64).to(device)
    criterion = nn.MSELoss() 
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 4. The Training Loop
    print("\nStarting Training...")
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        
        for batch_features, batch_targets in train_loader:
            # Move data to GPU
            batch_features = batch_features.to(device)
            batch_targets = batch_targets.to(device)
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass
            reconstruction = model(batch_features)
            
            # Calculate loss
            loss = criterion(reconstruction, batch_targets)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        # Calculate average loss for the epoch
        avg_train_loss = train_loss / len(train_loader)
        
        # 5. Validation Phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for val_features, val_targets in val_loader:
                val_features = val_features.to(device)
                val_targets = val_targets.to(device)
                
                val_recon = model(val_features)
                v_loss = criterion(val_recon, val_targets)
                val_loss += v_loss.item()
                
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Epoch [{epoch+1}/{EPOCHS}] | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")

    # 6. Save the trained model weights
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)
    save_path = model_dir / "lstm_autoencoder.pth"
    torch.save(model.state_dict(), save_path)
    print(f"\nModel saved successfully to {save_path}")
    
    return model, scaler

if __name__ == "__main__":
    train_autoencoder()