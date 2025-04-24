import numpy as np

class BicycleModel:
    def __init__(self, dt=0.05):
        self.L = 2.9
        self.dt = dt
        self.x = np.zeros(4)  # [x, y, yaw, v]

    def step(self, a, delta):
        x, y, yaw, v = self.x
        x += v * np.cos(yaw) * self.dt
        y += v * np.sin(yaw) * self.dt
        yaw += (v / self.L) * np.tan(delta) * self.dt
        v += a * self.dt
        self.x = np.array([x, y, yaw, v])
        return self.x

    def get_measurement(self):
        # 측정 가능한 값 (예: GPS 측정)
        noise = np.random.normal(0, 0.01, size=2)  # GPS 노이즈
        return self.x[:2] + noise  # x, y
