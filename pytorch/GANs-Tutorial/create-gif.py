import os
import imageio

def create_gif(png_dir, gif_path, duration=0.5, loop=0):
    images = []

    for file_name in sorted(os.listdir(png_dir)):
        if file_name.endswith('.png'):
            file_path = os.path.join(png_dir, file_name)
            images.append(imageio.imread(file_path))
    imageio.mimsave(gif_path, images, duration=duration, loop=loop)

png_dir_1 = 'Lab1-VanillaGAN/result/1-4/sample'
gif_path_1 = 'Lab1-VanillaGAN/figures/result_1-4.gif'

# png_dir_2 = 'Lab1-VanillaGAN/result/2-2/sample'
# gif_path_2 = 'Lab1-VanillaGAN/figures/result_2-2.gif'

create_gif(png_dir_1, gif_path_1)
# create_gif(png_dir_2, gif_path_2)