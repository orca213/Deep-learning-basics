import os
import numpy as np
from PIL import Image

def generate_gif(frames, output_dir="gifs", gif_name="demo.gif", duration=30):
    assert gif_name.endswith("gif"), "File name should end with .gif"
    os.makedirs(output_dir, exist_ok=True)
    gif_name = os.path.join(output_dir, gif_name)
    
    imgs = [Image.fromarray(img) for img in frames]
    imgs[0].save(gif_name, save_all=True, append_images=imgs[1:], duration=duration, loop=0)
    print(f"Render successful: [ {gif_name} ]")
    
    
def wrap_to_pi(x: float) -> float:
    """Wrap the input radian to (-pi, pi]. Note that -pi is exclusive and +pi is inclusive.

    Args:
        x (float): radian.

    Returns:
        The radian in range (-pi, pi].
    """
    angles = x
    angles %= 2 * np.pi
    angles -= 2 * np.pi * (angles > np.pi)
    return angles

def get_heading_diff(env):
    current_lane = env.vehicle.lane
    long_now, _ = current_lane.local_coordinates(env.vehicle.position)
    lane_heading = current_lane.heading_theta_at(long_now+1)
    vehicle_heading = env.vehicle.heading_theta
    heading_diff = wrap_to_pi(vehicle_heading-lane_heading)
    return heading_diff, lane_heading, vehicle_heading