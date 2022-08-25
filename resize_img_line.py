import os
import numpy as np
from PIL import Image, ImageDraw
import pandas as pd

def resize_img_line(img_path, line_path, resized_dim):
    line = open(line_path,'r').read().split(',')[:4]
    line = [int(x) for x in line]
    img = Image.open(img_path)
    shape = img.size
    factor_x = shape[0]
    factor_y = shape[1]
    img_resize = img.resize(resized_dim)
    # img_resize = np.array(img_resize)
    # img_resize_nor = img_resize / 127.5
    # img_resize_nor -= 1
    # img_draw = ImageDraw.Draw(img)
    # img_draw.line([(line[0]*resized_dim[0]/factor_x, line[1]*resized_dim[1]/factor_y),
    #                (line[2]*resized_dim[0]/factor_x, line[3]*resized_dim[1]/factor_y)], fill="red", width=3)
    line_resize = [(line[0]*resized_dim[0]/factor_x, line[1]*resized_dim[1]/factor_y),
                    (line[2]*resized_dim[0]/factor_x, line[3]*resized_dim[1]/factor_y)]
                   # (line[4]*resized_dim[0]/factor_x, line[5]*resized_dim[1]/factor_y)]
    return img_resize, line_resize


img_path = 'G:\Python projects\Horizon_Detection\data\img_data/'
line_path = 'G:\Python projects\Horizon_Detection\data\line_data/'
save_resized_img_path = 'G:\Python projects\Horizon_Detection\data/resized_img_data/'
save_resized_line_path = 'G:\Python projects\Horizon_Detection\data/resized_line_data/'
os.makedirs(save_resized_img_path)
os.makedirs(save_resized_line_path)
resized_dim = (224,224)
lines_y1_list = []
lines_y2_list = []
for i in range(1,954):
    img, line = resize_img_line(img_path+f'{i}.jpg', line_path+f'{i}.xml', resized_dim)
    name = format(i, '04')
    img.save(save_resized_img_path+f'{name}.jpg')
    with open(save_resized_line_path+f'{name}.txt','w') as f:
        f.write(f'{line[0][1]},{line[1][1]}')