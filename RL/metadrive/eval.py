from metadrive.envs.metadrive_env import MetaDriveEnv
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from tqdm import tqdm
from utils import generate_gif

if __name__ == "__main__":
    
    env = MetaDriveEnv(config={"use_render": False})
    model = PPO.load("ppo_policy_1000")

    # 환경 재설정 및 프레임 수집
    frames = []
    obs, _ = env.reset()
    for _ in tqdm(range(1000)):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        frame = env.render(mode="topdown", screen_record=True, window=False)
        frames.append(frame)
        if terminated or truncated:
            obs, _ = env.reset()
    
    env.close()

    # GIF 생성
    print("Rendering gif . . .", end="\t")
    generate_gif(frames, gif_name="demo_1000.gif")
    print("Done!")