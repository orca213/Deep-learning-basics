from metadrive.envs.metadrive_env import MetaDriveEnv
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from tqdm import tqdm
from utils import generate_gif

if __name__ == "__main__":
    # 환경 초기화
    env = MetaDriveEnv(config={"use_render": False})

    # 환경 유효성 검사 (선택 사항)
    check_env(env)

    # PPO 모델 초기화
    model = PPO("MlpPolicy", env, verbose=1)

    # 모델 학습
    print("Training PPO model...")
    model.learn(total_timesteps=30000)
    model.save("ppo_policy_1000")

    print("Done!")