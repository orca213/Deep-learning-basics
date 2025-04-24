import argparse
import numpy as np
from bicycle_model import BicycleModel
from kalman_filter import KalmanFilter
from controllers.lqr_controller import LQRController
from controllers.mlp_controller import MLControllerWrapper
from utils import generate_straight_trajectory, plot_results, compare_results

def get_kalman_matrices(dt=0.05):
    A = np.eye(4)
    A[0, 3] = dt  # x += v * dt
    A[1, 3] = dt  # y += v * dt
    B = np.zeros((4, 2))
    B[3, 0] = dt  # 가속도가 속도에 영향
    B[2, 1] = dt  # 조향이 yaw에 영향 (단순화)

    C = np.zeros((2, 4))
    C[0, 0] = 1  # 측정: x
    C[1, 1] = 1  # 측정: y

    Q = np.diag([0.2, 0.2, 0.2, 0.2])
    R = np.diag([0.05, 0.05])

    return A, B, C, Q, R

def run_simulation(controller_type="lqr", steps=100, dt=0.1):
    ref_traj = generate_straight_trajectory(steps, v_target=2.0, dt=dt)
    A, B, C, Q, R = get_kalman_matrices(dt)
    x0 = ref_traj[0] + np.random.normal(0, 0.2, size=(4,))
    P0 = np.eye(4)

    model = BicycleModel(dt=dt)
    kf = KalmanFilter(A, B, C, Q, R, P0, x0)

    if controller_type == "lqr":
        Q_lqr = np.diag([0.1, 0.1, 0.1, 0.1])
        R_lqr = np.diag([0.5, 0.5])
        controller = LQRController(A, B)
    elif controller_type == "ml":
        controller = MLControllerWrapper()
    else:
        raise ValueError("Invalid controller type")

    true_log = []
    est_log = []
    ref_log = []
    u_log = []

    for t in range(steps):
        y = model.get_measurement()
        x_hat = kf.estimate(np.zeros((2, 1)), y)
        x_ref = ref_traj[t]

        if controller_type == "lqr":
            u = controller.control(x_hat, x_ref)
        else:
            u = controller.control(x_hat)

        a = float(np.clip(u[0], -2.0, 2.0))
        delta = float(np.clip(u[1], -0.3, 0.3))

        true_x = model.step(a, delta)
        kf.predict(np.array([[a], [delta]]))

        true_log.append(true_x)
        est_log.append(x_hat)
        ref_log.append(x_ref)
        u_log.append([a, delta])

    return np.array(true_log), np.array(est_log), np.array(ref_log), np.array(u_log)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ctrl", type=str, choices=["lqr", "ml", "both"], default="lqr", help="Controller type: 'lqr', 'ml', or 'both'")
    args = parser.parse_args()

    if args.ctrl == "both":
        print("[▶] Running LQR controller...")
        lqr_true, _, ref, lqr_u = run_simulation("lqr")
        print("[▶] Running ML controller...")
        ml_true, _, _, ml_u = run_simulation("ml")
        compare_results(ref, lqr_true, lqr_u, ml_true, ml_u)
    else:
        true_log, est_log, ref_log, _ = run_simulation(args.ctrl)
        plot_results(true_log, est_log, ref_log)
    print("[+] Simulation completed.")