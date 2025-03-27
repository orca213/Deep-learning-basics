from metadrive.envs.metadrive_env import MetaDriveEnv
from env.env import MyEnv
from env.config import CONFIG
from stable_baselines3 import PPO
from tqdm import tqdm
from utils import generate_gif

if __name__ == "__main__":
    
    # env = MetaDriveEnv(config={"use_render": False})
    env = MyEnv(CONFIG)
    model = PPO.load("models/ppo_policy")

    # 환경 재설정 및 프레임 수집
    frames = []
    obs, _ = env.reset()
    reward_log, speed_log, driving_log = [], [], []
    total_reward = 0
    for _ in tqdm(range(1000), desc="Evaluating", unit="steps"):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        speed_log.append(env.vehicle.speed)
        driving_log.append(info["driving_reward"])
        frame = env.render(mode="topdown", screen_record=True, window=False)
        frames.append(frame)
        if terminated or truncated:
            obs, _ = env.reset()
            reward_log.append(total_reward)
            total_reward = 0
    
    env.close()
    
    print(f"Average reward: {sum(reward_log) / len(reward_log):.4f}", end="\t")
    print(f"Average speed: {sum(speed_log) / len(speed_log):.2f} km/h")
    print(f"Average driving reward: {sum(driving_log) / len(driving_log):.4f} m/step")

    # GIF 생성
    print("Rendering gif . . .", end="\t")
    generate_gif(frames, gif_name="demo.gif")
    print("Done!")