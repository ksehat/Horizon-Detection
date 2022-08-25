import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
import segmentation_models as sm

model = keras.models.load_model('G:\Python projects\Horizon_Detection\model.h5',
                                custom_objects={'binary_crossentropy_plus_jaccard_loss':sm.losses.bce_jaccard_loss,
                                                'iou_score':sm.metrics.iou_score})
# model.compile('Adam',
#               loss=sm.losses.bce_jaccard_loss,
#               metrics=[sm.metrics.iou_score])
seed = 10
batch_size = 10

img_data_gen_args = dict(
    rotation_range=90,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='reflect')

mask_data_gen_args = dict(
    rotation_range=90,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='reflect',
    preprocessing_function=lambda x: np.where(x < 60, 0, 1).astype(x.dtype))

image_data_generator = ImageDataGenerator(**img_data_gen_args)
mask_data_generator = ImageDataGenerator(**mask_data_gen_args)

test_img_generator = image_data_generator.flow_from_directory(
    'G:\Python projects\Horizon_Detection\data/train_test_x/test/', seed=seed, target_size=(224, 224),
    batch_size=batch_size, class_mode=None)  # x_test
test_mask_generator = mask_data_generator.flow_from_directory(
    'G:\Python projects\Horizon_Detection\data/train_test_y/test/', seed=seed, target_size=(224, 224),
    batch_size=batch_size, class_mode=None, color_mode='grayscale')  # y_test

test_img_batch = test_img_generator.next()
test_mask_batch = test_mask_generator.next()
pred_mask_batch = np.where(model.predict(test_img_batch) < .5, 0, 1)
for i in range(0, batch_size):
    image = test_img_batch[i]
    mask = pred_mask_batch[i]
    # plt.subplot(1, 2, i + 1)
    plt.imshow(image[:, :, 0])
    plt.show()
    # plt.subplot(1, 2, i + 1)
    plt.imshow(mask[:, :, 0])
    plt.show()
