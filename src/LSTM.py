import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib

# ----------------------------
# Load data
# ----------------------------
df = pd.read_csv("data/climate_features.csv")

series = df["T2M"].values.reshape(-1, 1)

# Scale data
scaler = MinMaxScaler()
series = scaler.fit_transform(series)


# ----------------------------
# Create sequences
# ----------------------------
def create_sequences(data, seq_len=30):
    X = []
    y = []

    for i in range(len(data) - seq_len):
        X.append(data[i:i + seq_len])
        y.append(data[i + seq_len])

    return np.array(X), np.array(y)


X, y = create_sequences(series)

X = torch.tensor(X).float()
y = torch.tensor(y).float()


# ----------------------------
# LSTM Model
# ----------------------------
class LSTMModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.lstm = nn.LSTM(input_size=1, hidden_size=32, batch_first=True)
        self.fc = nn.Linear(32, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out


model = LSTMModel()


# ----------------------------
# Training setup
# ----------------------------
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# ----------------------------
# Training loop
# ----------------------------
epochs = 10

for epoch in range(epochs):

    pred = model(X)

    loss = criterion(pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")


# ----------------------------
# Save model
# ----------------------------
torch.save(model.state_dict(), "models/lstm_model.pth")

print("Model saved successfully!")