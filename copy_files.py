import os
import shutil

for folder in os.listdir('G:\Python projects\Horizon_Detection\data'):
    for file in os.listdir(f'G:\Python projects\Horizon_Detection\data\{folder}'):
        if file.split('.')[-1] in ['xml']:
            shutil.copyfile(f'G:\Python projects\Horizon_Detection\data\{folder}/{file}', f'G:\Python projects\Horizon_Detection\data/{file}')