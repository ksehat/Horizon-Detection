import os

import numpy as np
import tensorflow as tf
import segmentation_models as sm
from segmentation_models import get_preprocessing
# from stacking_images import stack_images
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
import random
import splitfolders

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def main():
    sm.set_framework('tf.keras')
    sm.framework()

    gpus = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)

    # images = stack_images('G:\Python projects\Horizon_Detection\data/resized_img_data/', [400, 400], True, 3)
    # masks = stack_images('G:\Python projects\Horizon_Detection\data/resized_mask_data/', [400, 400], True, 1)
    # masks = np.floor(masks)

    # preprocess_input = sm.get_preprocessing('resnet50')
    # images1 = preprocess_input(images)
    seed = 24
    batch_size = 10

    # splitfolders.ratio('G:\Python projects\Horizon_Detection\data/x_data/',
    #                    output='G:\Python projects\Horizon_Detection\data/train_test_x/', seed=1, ratio=(.9, 0.1, 0))
    # splitfolders.ratio('G:\Python projects\Horizon_Detection\data/y_data/',
    #                    output='G:\Python projects\Horizon_Detection\data/train_test_y/', seed=1, ratio=(.9, 0.1, 0))

    # Sanity check
    # import random
    # img_num = random.randint(0, len(x_train))
    # plt.figure(figsize = (12,6))
    # plt.subplot(121)
    # plt.imshow(x_train[img_num,:,:,0])
    # plt.subplot(122)
    # plt.imshow(y_train[img_num,:,:,0])
    # plt.show()
    # endregion

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
        preprocessing_function=lambda x: np.where(x < 60, 0, 1).astype(x.dtype)) # actually we are doing the normalization here and devide into two classes of 0 and 1s

    image_data_generator = ImageDataGenerator(**img_data_gen_args)
    # image_data_generator.fit('G:\Python projects\Horizon_Detection\data/train_test_data/train/resized_img_data', augment=True, seed=seed)
    image_generator = image_data_generator.flow_from_directory(
        'G:\Python projects\Horizon_Detection\data/train_test_x/train/', seed=seed, target_size=(224, 224),
        batch_size=batch_size, class_mode=None)  # x_train
    valid_img_generator = image_data_generator.flow_from_directory(
        'G:\Python projects\Horizon_Detection\data/train_test_x/val/', seed=seed, target_size=(224, 224),
        batch_size=batch_size, class_mode=None)  # x_test
    test_img_generator = image_data_generator.flow_from_directory(
        'G:\Python projects\Horizon_Detection\data/train_test_x/test/', seed=seed, target_size=(224, 224),
        batch_size=batch_size, class_mode=None)  # x_test

    mask_data_generator = ImageDataGenerator(**mask_data_gen_args)
    # mask_data_generator.fit('G:\Python projects\Horizon_Detection\data/train_test_data/train/resized_mask_data', augment=True, seed=seed)
    mask_generator = mask_data_generator.flow_from_directory(
        'G:\Python projects\Horizon_Detection\data/train_test_y/train/', seed=seed, target_size=(224, 224),
        batch_size=batch_size, class_mode=None, color_mode='grayscale')  # y_train
    valid_mask_generator = mask_data_generator.flow_from_directory(
        'G:\Python projects\Horizon_Detection\data/train_test_y/val/', seed=seed, target_size=(224, 224),
        batch_size=batch_size, class_mode=None, color_mode='grayscale')  # y_test
    test_mask_generator = mask_data_generator.flow_from_directory(
        'G:\Python projects\Horizon_Detection\data/train_test_y/test/', seed=seed, target_size=(224, 224),
        batch_size=batch_size, class_mode=None, color_mode='grayscale')  # y_test

    def my_img_mask_generator(image_generator, mask_generator):
        train_generator = zip(image_generator, mask_generator)
        for (img, mask) in train_generator:
            yield (img, mask)

    # preprocess_input = get_preprocessing('resnet34')
    my_generator = my_img_mask_generator(image_generator, mask_generator)
    # my_generator = preprocess_input(my_generator)
    valid_datagen = my_img_mask_generator(valid_img_generator, valid_mask_generator)
    # valid_datagen = preprocess_input(valid_datagen)
    test_datagen = my_img_mask_generator(test_img_generator, test_mask_generator)

    x = image_generator.next()
    y = mask_generator.next()
    for i in range(0, 1):
        image = x[i]
        mask = y[i]
        plt.subplot(1, 2, 1)
        plt.imshow(image[:, :, 0])
        plt.subplot(1, 2, 2)
        plt.imshow(mask[:, :, 0])
        plt.show()

    model = sm.Unet('resnet50', encoder_weights='imagenet')
    model.compile('Adam',
                  loss=sm.losses.bce_jaccard_loss,
                  metrics=[sm.metrics.iou_score])

    batch_size = batch_size
    len_x_train = len(os.listdir('G:\Python projects\Horizon_Detection\data/train_test_x/train/resized_img_data/'))
    steps_per_epoch = 3 * (len_x_train) // batch_size

    callback = tf.keras.callbacks.EarlyStopping(
        monitor='iou_score',
        patience=5,
        verbose=0,
        mode='max',
        baseline=None,
        restore_best_weights=True
    )
    class_weight = {0: 1.,
                    1: 50.}
    history = model.fit(my_generator,
                        validation_data=valid_datagen,
                        steps_per_epoch=steps_per_epoch,
                        validation_steps=steps_per_epoch,
                        callbacks=[callback],
                        epochs=100)
                        # class_weight=class_weight)
    model.save('model.h5')
    # plot the training and validation accuracy and loss at each epoch
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, 'y', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    acc = history.history['iou_score']
    val_acc = history.history['val_iou_score']

    plt.plot(epochs, acc, 'y', label='Training acc')
    plt.plot(epochs, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    # IOU
    # y_pred = model.predict(test_img_generator)
    # y_pred_thresholded = y_pred > 0.5
    #
    # intersection = np.logical_and(test_mask_generator, y_pred_thresholded)
    # union = np.logical_or(test_mask_generator, y_pred_thresholded)
    # iou_score = np.sum(intersection) / np.sum(union)
    # print("IoU socre is: ", iou_score)

    # Predict on a few images
    # model = get_model()
    # model.load_weights('mitochondria_50_plus_100_epochs.hdf5') #Trained for 50 epochs and then additional 100

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

    # plt.figure(figsize=(16, 8))
    # plt.subplot(231)
    # plt.title('Testing Image')
    # plt.imshow(test_img[:, :, 0], cmap='gray')
    # plt.subplot(232)
    # plt.title('Testing Label')
    # plt.imshow(ground_truth[:, :, 0], cmap='gray')
    # plt.subplot(233)
    # plt.title('Prediction on test image')
    # plt.imshow(prediction, cmap='gray')
    #
    # plt.show()


if __name__ == '__main__':
    main()
