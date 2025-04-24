import torch
import torch.nn as nn
import numpy as np

# MLP 구조는 학습 때와 동일해야 함
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

class MLControllerWrapper:
    def __init__(self, model_path="mlp_controller.pth", scaler_x_path="scaler_x_mean_std.npy", scaler_y_path="scaler_y_mean_std.npy"):
        # models/mlp_controller.pth
        model_path = 'models/' + model_path
        scaler_x_path = 'models/' + scaler_x_path
        scaler_y_path = 'models/' + scaler_y_path
        
        # 모델 불러오기
        self.model = MLPController()
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

        # 스케일러 로딩
        scaler_x = np.load(scaler_x_path, allow_pickle=True)
        scaler_y = np.load(scaler_y_path, allow_pickle=True)
        self.x_mean, self.x_std = scaler_x
        self.y_mean, self.y_std = scaler_y

    def control(self, x_hat):
        # 입력 정규화
        x_norm = (x_hat - self.x_mean) / self.x_std
        x_tensor = torch.tensor(x_norm, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            u_norm = self.model(x_tensor).squeeze(0).numpy()
        # 출력 역변환
        u = u_norm * self.y_std + self.y_mean
        return float(u[0]), float(u[1])  # (a, delta)
