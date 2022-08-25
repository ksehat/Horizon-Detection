import os
import math
import numpy as np
import numpy.ma as ma
import pandas as pd
from PIL import Image, ImageOps
from skimage.draw import line
import json

mode = 'multi-line'
if mode != 'multi-line':
    data_dir = 'G:\Python projects\Horizon_Detection\data/resized_img_data/'
    if not os.path.isdir('G:\Python projects\Horizon_Detection\data/resized_mask_data/'):
        os.makedirs('G:\Python projects\Horizon_Detection\data/resized_mask_data/')
    for file in os.listdir(data_dir):
        if file.split('.')[-1] == 'jpg':
            line_file = 'G:\Python projects\Horizon_Detection\data/resized_line_data/' + file.split('.')[0] + '.txt'
            line_data = open(line_file, 'r').read().split(',')
            img = Image.open(data_dir + file)
            # img = ImageOps.grayscale(img)
            image = np.array(img)
            rr, cc = line(math.floor(float(line_data[0])), 0, math.floor(float(line_data[1])), 223)
            image[image != image[rr, cc, 0]] = 0
            image[image != image[rr, cc, 1]] = 0
            image[image != image[rr, cc, 2]] = 0
            image[rr, cc, 0] = 255
            image[rr, cc, 1] = 255
            image[rr, cc, 2] = 255
            image[image != 255] = 0
            pil_image = Image.fromarray(image)
            # pil_image = ImageOps.grayscale(pil_image)
            # pil_image.show()

            pil_image.save('G:\Python projects\Horizon_Detection\data/resized_mask_data/' + file.split('.')[0] + '.jpg')

if mode == 'multi-line':
    resized_dim = (224, 224)
    base_dir = 'G:\Python projects\Horizon_Detection\data\my_data/'
    df = pd.read_csv(base_dir + 'df.csv')
    for idx, file in enumerate(os.listdir(base_dir)):
        if 'PNG' in file.split('.'):
            img = Image.open(base_dir + file)
            rgb_img = img.convert('RGB')
            shape = rgb_img.size
            factor_x = shape[0]
            factor_y = shape[1]
            img_resize = rgb_img.resize(resized_dim)
            image = np.array(img_resize)
            line_dict = json.loads(df[df['filename'] == file].reset_index().loc[0, 'region_shape_attributes'])
            x_list = line_dict['all_points_x']
            y_list = line_dict['all_points_y']
            x_list_resized = [math.floor(x * resized_dim[0] / factor_x) for x in x_list]
            y_list_resized = [math.floor(y * resized_dim[1] / factor_y) for y in y_list]
            # img = ImageOps.grayscale(img)

            # image=np.expand_dims(image, -1)
            for i in range(len(x_list_resized)):
                if i < len(x_list_resized) - 1:
                    rr, cc = line(math.floor(float(y_list_resized[i])),
                                  math.floor(float(x_list_resized[i])),
                                  math.floor(float(y_list_resized[i + 1])),
                                  math.floor(float(x_list_resized[i + 1])))
                    if i == 0:
                        image = np.zeros((image.shape[0], image.shape[1], 3))
                    image[rr, cc] = 255
            pil_image = Image.fromarray(image.astype(np.uint8))
            # pil_image.show()
            if not os.path.isdir('G:\Python projects\Horizon_Detection\data/resized_mask_my_data/'):
                os.makedirs('G:\Python projects\Horizon_Detection\data/resized_mask_my_data/')
            pil_image.save(
                'G:\Python projects\Horizon_Detection\data/resized_mask_data/' + format(idx + 954, '04') + '.jpg')
            img_resize.save(
                'G:\Python projects\Horizon_Detection\data/resized_img_data/' + format(idx + 954, '04') + '.jpg')