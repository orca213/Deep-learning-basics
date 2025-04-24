import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def plot_results(true_log, est_log, ref_log):
    plt.figure()
    plt.plot(true_log[:, 0], true_log[:, 1], label="True")
    plt.plot(est_log[:, 0], est_log[:, 1], '--', label="Estimated")
    plt.plot(ref_log[:, 0], ref_log[:, 1], ':', label="Reference")
    plt.xlabel("X [m]")
    plt.ylabel("Y [m]")
    plt.title("Trajectory Tracking with Kalman Filter + LQR")
    plt.legend()
    plt.axis("equal")
    plt.grid()
    plt.savefig("sim_out.png")

def generate_straight_trajectory(steps, v_target=2.0, dt=0.1):
    traj = []
    for t in range(steps):
        x = v_target * dt * t
        y = 0.0
        yaw = 0.0
        v = v_target
        traj.append(np.array([x, y, yaw, v]))
    return np.array(traj)

def generate_sine_trajectory(steps, v_target=2.0, dt=0.1, amplitude=2.0, frequency=0.1):
    traj = []
    for t in range(steps):
        x = v_target * dt * t
        y = amplitude * np.sin(frequency * x)
        yaw = np.arctan2(amplitude * frequency * np.cos(frequency * x), 1.0)
        v = v_target
        traj.append(np.array([x, y, yaw, v]))
    return np.array(traj)

def compute_rmse(true_xy, ref_xy):
    error = true_xy - ref_xy
    mse = np.mean(np.sum(error ** 2, axis=1))
    return np.sqrt(mse)

def compute_smoothness(u_seq):
    # u_seq: [N x 2] array (a, delta)
    diff = np.diff(u_seq, axis=0)
    return np.sum(diff ** 2)

def compare_results(ref, true_lqr, u_lqr, true_ml, u_ml):
    # RMSE 비교
    rmse_lqr = compute_rmse(true_lqr[:, :2], ref[:, :2])
    rmse_ml = compute_rmse(true_ml[:, :2], ref[:, :2])

    # 제어 smoothness 비교
    smooth_lqr = compute_smoothness(u_lqr)
    smooth_ml = compute_smoothness(u_ml)

    print("[RMSE]")
    print(f"LQR: {rmse_lqr:.4f} m")
    print(f"ML : {rmse_ml:.4f} m\n")

    print("[Smoothness (lower = smoother)]")
    print(f"LQR: {smooth_lqr:.4f}")
    print(f"ML : {smooth_ml:.4f}")

    # 시각화 비교
    plt.figure(figsize=(8, 6))
    plt.plot(ref[:, 0], ref[:, 1], 'k--', label="Reference")
    plt.plot(true_lqr[:, 0], true_lqr[:, 1], 'g', label="LQR")
    plt.plot(true_ml[:, 0], true_ml[:, 1], 'b', label="MLP")
    plt.legend()
    plt.xlabel("X [m]")
    plt.ylabel("Y [m]")
    plt.title("Trajectory Comparison")
    plt.grid()
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig("sim_out.png")