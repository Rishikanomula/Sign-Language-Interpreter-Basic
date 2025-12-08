# models/lstm_model.py
import torch
import torch.nn as nn

class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, bidirectional=False, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, 
                            dropout=dropout, bidirectional=bidirectional)
        direction = 2 if bidirectional else 1
        self.fc = nn.Sequential(
            nn.Linear(hidden_size*direction, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        # x: (B, seq_len, feat)
        out, (hn, cn) = self.lstm(x)   # out: (B, seq_len, hidden*dir)
        # take last timestep
        last = out[:, -1, :]
        return self.fc(last)
