import numpy as np

class KalmanFilter:
    def __init__(self, A, B, C, Q, R, P0, x0):
        self.A = A
        self.B = B
        self.C = C
        self.Q = Q
        self.R = R
        self.P = P0
        self.x_hat = x0.reshape(-1, 1)  # (n, 1)

    def predict(self, u):
        u = u.reshape(-1, 1)  # Ensure shape (m, 1)
        self.x_hat = self.A @ self.x_hat + self.B @ u
        self.P = self.A @ self.P @ self.A.T + self.Q

    def update(self, y):
        y = y.reshape(-1, 1)  # (p, 1)
        y_hat = self.C @ self.x_hat  # (p, 1)
        innovation = y - y_hat       # (p, 1)
        S = self.C @ self.P @ self.C.T + self.R
        K = self.P @ self.C.T @ np.linalg.inv(S)
        self.x_hat += K @ innovation
        I = np.eye(self.A.shape[0])
        self.P = (I - K @ self.C) @ self.P

    def estimate(self, u, y):
        self.predict(u)
        self.update(y)
        return self.x_hat.flatten()  # Return as (n,)
