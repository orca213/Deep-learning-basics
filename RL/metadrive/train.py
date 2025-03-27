from metadrive.envs.metadrive_env import MetaDriveEnv
from env.env import MyEnv
from env.config import CONFIG
from stable_baselines3 import PPO
import os, time


if __name__ == "__main__":
    # env = MetaDriveEnv(config={"use_render": False})
    env = MyEnv(CONFIG)

    # PPO 모델 초기화
    model = PPO("MlpPolicy", env, verbose=1)

    # 모델 학습
    print("Training PPO model...")
    start = time.time()
    model.learn(total_timesteps=50000)
    
    # 학습된 모델 저장
    os.makedirs("models", exist_ok=True)
    model.save("models/ppo_policy")

    print("Done!")
    print(f"Training time: {time.time() - start:.2f} seconds")