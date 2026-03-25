import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

class LOBDataset(Dataset):
    def __init__(self, dataframe, seq_len=60, scaler=None, is_train=True):
        self.seq_len = seq_len
        raw_data = dataframe.values
        if is_train:
            self.scaler = StandardScaler()
            self.data = self.scaler.fit_transform(raw_data)
        else:
            self.scaler = scaler
            self.data = self.scaler.transform(raw_data)
            
        self.data = torch.FloatTensor(self.data)

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        window = self.data[idx : idx + self.seq_len]
        return window, window

def create_dataloaders(train_df, test_df, seq_len=60, batch_size=256):
    train_dataset = LOBDataset(train_df, seq_len=seq_len, is_train=True)
    test_dataset = LOBDataset(test_df, seq_len=seq_len, scaler=train_dataset.scaler, is_train=False)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    
    return train_loader, test_loader, train_dataset.scaler