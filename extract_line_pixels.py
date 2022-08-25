import os
import math
import numpy as np
from PIL import Image
from skimage.draw import line

data_dir = 'G:\Python projects\Horizon_Detection\data/resized_img_data/'
for file in os.listdir(data_dir):
    if file.split('.')[-1] == 'jpg':
        line_file = file.split('.')[0]+'.txt'
        line_data = open(data_dir+line_file,'r').read().split(',')
        img = Image.open(data_dir+file)
        image = np.array(img)
        rr, cc = line(math.floor(float(line_data[0])),0, math.floor(float(line_data[1])), 399)
        i=0
        for x in rr:
            if i==0:
                open(data_dir + line_file, 'w').write(str(x)+',')
            else:
                open(data_dir + line_file, 'a').write(str(x) + ',')
            i += 1
        # pil_image = Image.fromarray(image)
        # pil_image.show()




