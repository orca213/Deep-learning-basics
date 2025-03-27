from PIL import Image

def generate_gif(frames, gif_name="demo.gif", duration=30):
    assert gif_name.endswith("gif"), "File name should end with .gif"
    imgs = [Image.fromarray(img) for img in frames]
    imgs[0].save(gif_name, save_all=True, append_images=imgs[1:], duration=duration, loop=0)
    print(f"Render successful: [ {gif_name} ]")