import time
import urllib.request
import os
import zipfile
import random
import numpy as np
import pandas as pd
from PIL import Image,ImageDraw
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.utils import plot_model
from tensorflow.keras.utils import image_dataset_from_directory
import matplotlib.pyplot as plt

def dataset_show(data_set):
    plt.figure(figsize=(10, 10))
    for images in data_set.take(1):
      for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.show()

def get_line(file_path):
    # Convert the path to a list of path components
    l1 = []
    tensor_file_name = tf.strings.split(file_path, '.')[0] + '.txt'
    for i in range(400):
        l1.append(float(tf.strings.split(tf.io.read_file(tensor_file_name),',')[i]))
    # l1.append(float(tf.strings.split(tf.io.read_file(file),',')[1]))
    # return tf.convert_to_tensor(l1)
    return tf.convert_to_tensor(l1, dtype=tf.float32)

def decode_img(img):
    # Convert the compressed string to a 3D uint8 tensor
    img = tf.io.decode_jpeg(img, channels=3)
    # img = tf.image.resize(img, [299, 299])
    img = img/255
    # Resize the image to the desired size
    return tf.image.resize(img, img_size)

def process_path(file_path):
    line = get_line(file_path)
    # Load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    return img, line

def process_path_for_test(file_path):
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    return tf.image.resize(img, img_size)


img_size = [400,400]
resized_train_dir = 'G:\Python projects\Horizon_Detection\data/resized_img_data'
resized_valid_dir = 'G:\Python projects\Horizon_Detection\data/resized_line_data'
list_ds = tf.data.Dataset.list_files(str(resized_train_dir+'/*.jpg'),shuffle=False)
AUTOTUNE = tf.data.AUTOTUNE
ds = list_ds.map(process_path, num_parallel_calls=AUTOTUNE)
ds = ds.shuffle(953)
batch_size = 100
train_ds = ds.take(800).batch(batch_size)
val_ds = ds.skip(800).take(100).batch(batch_size)
test_ds = ds.skip(900).take(53)

# for x in train_ds.take(5):
#     print(x)
# for x in val_ds.take(5):
#     print(x)

# train_ds = train_ds.map(process_path, num_parallel_calls=AUTOTUNE).batch(20)
# val_ds = val_ds.map(process_path, num_parallel_calls=AUTOTUNE).batch(20)
# test_ds = test_ds.map(process_path_for_test, num_parallel_calls=AUTOTUNE)
# for image, label in train_ds.take(1):
#     print("Image shape: ", image.numpy().shape)
#     print("Label: ", label.numpy())

# os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin/'

# Instantiate the model
pre_trained_model = InceptionV3(
    include_top=False,
    weights='imagenet',
    input_tensor=None,
    input_shape=(img_size[0],img_size[1],3),
    pooling=None,
    classes=1000,
    classifier_activation='softmax'
)
# pre_trained_model = tf.keras.applications.resnet50.ResNet50(
#     include_top=True,
#     weights='imagenet',
#     input_tensor=None,
#     input_shape=None,
#     pooling=None,
#     classes=1000
# )
# freeze the layers
for layer in pre_trained_model.layers:
    layer.trainable = False
last_layer = pre_trained_model.get_layer('mixed7')
last_output = last_layer.output
# Flatten the output layer to 1 dimension
x = layers.Flatten()(last_output)
# Add a fully connected layer with 1,024 hidden units and ReLU activation
# x = layers.Dense(30, activation='relu')(x)
x = layers.Dense(img_size[0], activation='relu')(x)
model = Model(pre_trained_model.input, x)
model.compile(optimizer='adam', #RMSprop(learning_rate=0.01),
              loss='mse',
              metrics=['acc'])

# pre_trained_model.summary()
# plot_model(model, to_file='pre_trained_model.png', show_shapes=True, show_layer_names=True)

callback = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=6,
    verbose=0,
    mode='min',
    baseline=None,
    restore_best_weights=True
)
history = model.fit(
    train_ds,
    validation_data=val_ds,
    callbacks=[callback],
    epochs=100,
    verbose=1)
model.save('model.h5')
pred = model.predict(test_ds.batch(batch_size))
# print(pred)
img_test = []
line_test = []
i = 0
for x in test_ds.take(len(test_ds)):
    cc = list(range(img_size[0]))
    img_test_arr = np.array(x[0])*255
    img_test_arr[np.int_(np.floor(pred[i])), np.array(cc)] = 0
    line_test.append(np.array(x[1]))
    img = Image.fromarray(img_test_arr.astype('uint8'), 'RGB')

    # img1 = ImageDraw.Draw(img)
    # img1.line([(0,np.array(x[1])[0]),(img_size[0],np.array(x[1])[1])], fill="red", width=0)
    # img1.line([(0,pred[i,0]),(img_size[0],pred[i,1])], fill="green", width=0)
    img.save(f'G:\Python projects\Horizon_Detection\data/resized_test_data/{i}.jpg')
    i += 1



