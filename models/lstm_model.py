import torch
import torch.nn as nn
import numpy as np

class LSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(1, 32, batch_first=True)
        self.fc = nn.Linear(32, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1])

def predict_future(df, steps=20):
    prices = df["close"].values[-50:]
    trend = np.polyfit(range(len(prices)), prices, 1)
    future = [trend[0]*i + trend[1] for i in range(len(prices), len(prices)+steps)]
    return future
