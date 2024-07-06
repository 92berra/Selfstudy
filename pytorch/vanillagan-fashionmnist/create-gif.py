import os
import imageio

def create_gif(png_dir, gif_path, duration=0.5):
    images = []
    for file_name in sorted(os.listdir(png_dir)):
        if file_name.endswith('.png'):
            file_path = os.path.join(png_dir, file_name)
            images.append(imageio.imread(file_path))
    imageio.mimsave(gif_path, images, duration=duration)

png_dir = './result/sample'
gif_path = './result/sample_animation.gif'

create_gif(png_dir, gif_path)