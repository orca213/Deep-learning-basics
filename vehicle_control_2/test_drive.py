import numpy as np
from tqdm import tqdm

from config import CONFIG
from utils import generate_gif, get_heading_diff
from metadrive.envs.metadrive_env import MetaDriveEnv


if __name__ == "__main__":
    env = MetaDriveEnv(CONFIG)

    frames = []
    obs, info = env.reset()
    for i in tqdm(range(1000), desc="Test driving", unit="steps"):
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        frame = env.render(mode="topdown", 
                            screen_record=True,
                            window=False)
        frames.append(frame)
        if terminated or truncated:
            env.reset()
        
        # heading diff [-pi, pi]
        # heading_diff = get_heading_diff(env)
        
        print(obs.shape)
        print(type(obs))
        exit()
    env.close()

    print(f"Test drive successful! -- Number of time steps: {len(frames)}")
    print("Rendering gif . . .", end="\t")
    generate_gif(frames)