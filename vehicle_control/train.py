import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import numpy as np
import os

# === MLP 모델 정의 === #
class MLPController(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=64, output_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)

# === 데이터 로딩 및 전처리 === #
def load_data(path='data/lqr_data.csv'):
    df = pd.read_csv(path)
    X = df[['x', 'y', 'yaw', 'v']].values
    y = df[['a', 'delta']].values

    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    X = scaler_x.fit_transform(X)
    y = scaler_y.fit_transform(y)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    return torch.tensor(X_train, dtype=torch.float32), \
           torch.tensor(y_train, dtype=torch.float32), \
           torch.tensor(X_val, dtype=torch.float32), \
           torch.tensor(y_val, dtype=torch.float32), \
           scaler_x, scaler_y

# === 학습 함수 === #
def train(model, train_x, train_y, val_x, val_y, epochs=10000, lr=1e-3):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    pbar = tqdm(total=epochs, desc="Training", unit="epoch")
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(train_x)
        loss = criterion(output, train_y)
        loss.backward()
        optimizer.step()
        pbar.update(1)

        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                val_loss = criterion(model(val_x), val_y)
            pbar.set_postfix(train_loss=loss.item(), val_loss=val_loss.item())

    return model

# === 실행 === #
def main():
    train_x, train_y, val_x, val_y, scaler_x, scaler_y = load_data()
    model = MLPController()
    model = train(model, train_x, train_y, val_x, val_y)

    # 모델 및 스케일러 저장
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/mlp_controller.pth")
    np.save("models/scaler_x_mean_std.npy", [scaler_x.mean_, scaler_x.scale_])
    np.save("models/scaler_y_mean_std.npy", [scaler_y.mean_, scaler_y.scale_])
    print("[+] Model and scalers saved.")

if __name__ == "__main__":
    main()
