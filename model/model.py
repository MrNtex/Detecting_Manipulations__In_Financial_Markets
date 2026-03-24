import torch
import torch.nn as nn

class LSTMAutoencoder(nn.Module):
    def __init__(self, seq_len, n_features, embedding_dim=64):
        super(LSTMAutoencoder, self).__init__()
        
        self.seq_len = seq_len
        self.n_features = n_features
        self.embedding_dim = embedding_dim
        
        self.encoder_lstm1 = nn.LSTM(
            input_size=n_features, 
            hidden_size=128, 
            num_layers=1, 
            batch_first=True
        )
        self.encoder_lstm2 = nn.LSTM(
            input_size=128, 
            hidden_size=embedding_dim, 
            num_layers=1, 
            batch_first=True
        )
        self.decoder_lstm1 = nn.LSTM(
            input_size=embedding_dim, 
            hidden_size=128, 
            num_layers=1, 
            batch_first=True
        )
        self.decoder_lstm2 = nn.LSTM(
            input_size=128, 
            hidden_size=n_features, 
            num_layers=1,  
            batch_first=True
        )
        
        self.output_layer = nn.Linear(n_features, n_features)

    def forward(self, x):
        x, (_, _) = self.encoder_lstm1(x)
        x, (hidden_n, _) = self.encoder_lstm2(x)

        encoded = hidden_n.squeeze(0).unsqueeze(1).repeat(1, self.seq_len, 1)

        x, (_, _) = self.decoder_lstm1(encoded)
        x, (_, _) = self.decoder_lstm2(x)

        reconstruction = self.output_layer(x)
        
        return reconstruction