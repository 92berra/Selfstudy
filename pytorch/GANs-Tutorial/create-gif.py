import os
import imageio

def create_gif(png_dir, duration=0.5):
    images = []
    sum = 0

    for i, file_name in enumerate(sorted(os.listdir(png_dir))):
        if i % 10000 == 0 :
            sum = sum + 1
    print(sum)

    # for file_name in sorted(os.listdir(png_dir)):
    #     if file_name.endswith('.png'):
    #         file_path = os.path.join(png_dir, file_name)
    #         images.append(imageio.imread(file_path))
    # imageio.mimsave(gif_path, images, duration=duration)

png_dir = 'Lab1-VanillaGAN/result/2/sample'
#gif_path = 'Lab1-VanillaGAN/result/2/sample_animation.gif'

create_gif(png_dir)