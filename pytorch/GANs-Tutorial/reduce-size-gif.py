from PIL import Image, ImageSequence

def reduce_gif_size(input_path, output_path, resize_factor=1, color_reduction=256):

    img = Image.open(input_path)
    
    frames = []

    for frame in ImageSequence.Iterator(img):
        if resize_factor != 1:
            frame = frame.resize(
                (frame.width // resize_factor, frame.height // resize_factor), Image.LANCZOS)
        
        frame = frame.convert('P', palette=Image.ADAPTIVE, colors=color_reduction)
        
        frames.append(frame)
    
    frames[0].save(
        output_path, 
        save_all=True, 
        append_images=frames[1:], 
        optimize=True, 
        loop=0
    )

input_gif_path = 'Lab1-VanillaGAN/figures/result_1-4.gif'
output_gif_path = 'Lab1-VanillaGAN/figures/result_1-4_reduced.gif'
#resize_factor = 2 
resize_factor = 3
# color_reduction = 128
color_reduction = 64

reduce_gif_size(input_gif_path, output_gif_path, resize_factor, color_reduction)
