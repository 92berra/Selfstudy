import os
import imageio

def create_gif(png_dir, gif_path, duration=0.5):
    images = []

    for i in range(len(png_dir)):
        
        if i % 100 == 0:
            file_path = os.path.join(png_dir, file_name)
            images.append(imageio.imread(file_path))
    imageio.mimsave(gif_path, images, duration=duration)

    # for file_name in sorted(os.listdir(png_dir)):
    #     if file_name.endswith('.png'):
    #         file_path = os.path.join(png_dir, file_name)
    #         images.append(imageio.imread(file_path))
    # imageio.mimsave(gif_path, images, duration=duration)

png_dir = 'Lab1-VanillaGAN/result/2/sample'
gif_path = 'Lab1-VanillaGAN/result/2/sample_animation.gif'

create_gif(png_dir, gif_path)