import numpy as np
import pandas as pd
from bicycle_model import BicycleModel
from kalman_filter import KalmanFilter, get_kalman_matrices
from controllers.lqr_controller import LQRController
from utils import generate_straight_trajectory

def collect_lqr_data(save_path='lqr_data.csv', steps=2000, dt=0.1):
    # 시스템 초기화
    A, B, C, Q, R = get_kalman_matrices(dt)
    x0 = np.zeros(4)
    P0 = np.eye(4)

    model = BicycleModel(dt=dt)
    kf = KalmanFilter(A, B, C, Q, R, P0, x0)
    controller = LQRController(A, B)
    ref_traj = generate_straight_trajectory(steps, v_target=2.0, dt=dt)

    records = []

    for t in range(steps):
        # 측정 → 추정
        y = model.get_measurement()
        u_dummy = np.zeros((2, 1))  # 제어 없을 때 예측용
        x_hat = kf.estimate(u_dummy, y)

        # 레퍼런스 상태
        x_ref = ref_traj[t]

        # 제어 입력 계산
        u = controller.control(x_hat, x_ref)

        # 입력 saturation
        a = float(np.clip(u[0], -2.0, 2.0))
        delta = float(np.clip(u[1], -0.3, 0.3))

        # 차량 상태 갱신
        model.step(a, delta)
        kf.predict(np.array([[a], [delta]]))

        # 데이터 저장
        record = np.hstack((x_hat, [a, delta]))
        records.append(record)

    df = pd.DataFrame(records, columns=['x', 'y', 'yaw', 'v', 'a', 'delta'])
    df.to_csv(save_path, index=False)
    print(f"[+] Saved {steps} samples to {save_path}")

if __name__ == "__main__":
    collect_lqr_data()
