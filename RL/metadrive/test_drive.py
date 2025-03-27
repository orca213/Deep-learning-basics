from metadrive.envs.metadrive_env import MetaDriveEnv
from tqdm import tqdm
from utils import generate_gif

if __name__ == "__main__":
    env = MetaDriveEnv()

    frames = []
    obs, info = env.reset()
    for i in tqdm(range(1000)):
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        frame = env.render(mode="topdown", 
                            screen_record=True,
                            window=False)
        frames.append(frame)
        if terminated or truncated:
            env.reset()
    env.close()

    print(f"Test drive successful! -- Number of time steps: {len(frames)}")
    print("Rendering gif . . .", end="\t")
    generate_gif(frames)