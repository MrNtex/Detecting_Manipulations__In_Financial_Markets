import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

class LOBDataset(Dataset):
    """
    A memory-efficient PyTorch Dataset for LOB Time Series.
    It takes a 2D Pandas DataFrame and yields 3D rolling windows on the fly.
    """
    def __init__(self, dataframe, seq_len=60, scaler=None, is_train=True):
        self.seq_len = seq_len
        
        # We drop the timestamp index if it exists, we only want the raw numbers
        raw_data = dataframe.values
        
        # LSTMs require scaled data (mean=0, variance=1)
        if is_train:
            self.scaler = StandardScaler()
            self.data = self.scaler.fit_transform(raw_data)
        else:
            # For testing, we MUST use the scaler fitted on the training data
            self.scaler = scaler
            self.data = self.scaler.transform(raw_data)
            
        # Convert the scaled numpy array to a PyTorch tensor
        self.data = torch.FloatTensor(self.data)

    def __len__(self):
        # If we have 1000 rows and a window of 60, we can only make 940 windows
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        # This is the magic. It simply slices 60 rows on the fly.
        # Zero data duplication in RAM.
        window = self.data[idx : idx + self.seq_len]
        
        # For an Autoencoder, the input is the target! 
        # We feed it the window, and tell it to predict the exact same window.
        return window, window

def create_dataloaders(train_df, test_df, seq_len=60, batch_size=256):
    """
    Wraps the datasets in DataLoaders for efficient batching and multi-processing.
    """
    print("Building Training Dataset...")
    train_dataset = LOBDataset(train_df, seq_len=seq_len, is_train=True)
    
    print("Building Testing Dataset...")
    # Pass the scaler from the train set to the test set to prevent data leakage
    test_dataset = LOBDataset(test_df, seq_len=seq_len, scaler=train_dataset.scaler, is_train=False)
    
    # DataLoader handles the batching (e.g., passing 256 windows to the GPU at once)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    # We do NOT shuffle the test loader so we can plot anomalies in chronological order later
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    
    return train_loader, test_loader, train_dataset.scaler