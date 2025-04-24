import numpy as np
from scipy.linalg import solve_continuous_are

class LQRController:
    def __init__(self, A, B, Q=None, R=None):
        self.A = A
        self.B = B
        self.Q = Q if Q is not None else np.eye(A.shape[0])
        self.R = R if R is not None else np.eye(B.shape[1])
        self.K = self.compute_gain()

    def compute_gain(self):
        P = solve_continuous_are(self.A, self.B, self.Q, self.R)
        K = np.linalg.inv(self.R) @ self.B.T @ P
        return K

    def control(self, x_hat, x_ref):
        # 상태 오차 기반 제어 입력 계산
        u = -self.K @ (x_hat - x_ref)
        return u
