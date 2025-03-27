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
    total_reward = []
    for _ in tqdm(range(1000), desc="Evaluating", unit="steps"):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        frame = env.render(mode="topdown", screen_record=True, window=False)
        frames.append(frame)
        if terminated or truncated:
            obs, _ = env.reset()
    
    env.close()
    print(f"Evaluation done! -- Total reward: {total_reward}")

    # GIF 생성
    print("Rendering gif . . .", end="\t")
    generate_gif(frames, gif_name="demo.gif")
    print("Done!")