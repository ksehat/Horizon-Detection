import os
from PIL import Image
import numpy as np


def stack_images(images_path, dim_size, normalize, num_channels):
    files_list = [x for x in os.listdir(images_path) if ('.jpg' in x) or ('.png' in x)]
    image_stacks = np.zeros(shape=(len(files_list), dim_size[0], dim_size[1], num_channels))
    for i,img in enumerate(files_list):
        if img.split('.')[-1] == 'jpg' or img.split('.')[-1] == 'png':
            image = np.array(Image.open(images_path+img))
            if num_channels==1:
                image = np.expand_dims(image, -1)
            if normalize:
                image_stacks[i] = image/255
            else:
                image_stacks[i] = image
    return image_stacks