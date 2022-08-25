import cv2
import math
from collections import deque
from imutils.video import FPS
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
import segmentation_models as sm
from PIL import Image, ImageDraw
import tensorflow as tf
from numpy.polynomial.polynomial import polyfit
from skimage.draw import line


def frame_transform(frame):
    resized_frame = cv2.resize(frame, (224, 224), fx=0, fy=0,
                               interpolation=cv2.INTER_CUBIC)
    return resized_frame


if __name__ == "__main__":

    gpus = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)

    model = keras.models.load_model('G:\Python projects\Horizon_Detection\model.h5',
                                    custom_objects={'binary_crossentropy_plus_jaccard_loss': sm.losses.bce_jaccard_loss,
                                                    'iou_score': sm.metrics.iou_score})
    path_vid = 'G:\Python projects\Horizon_Detection/video_data/1.mp4'
    video_capture = cv2.VideoCapture(path_vid)

    while True:
        success, frame_org = video_capture.read()
        if success:
            frame = frame_transform(frame_org)
            pred = model.predict(np.expand_dims(frame, 0))[0]
            mask = (np.where(pred < .98, False, True)).squeeze()

            # x_train = np.where(mask == True)[0]
            # y_train = np.where(mask == True)[1]
            # p = np.polyfit(x_train, y_train, 1)
            # m = p[0]
            # b = p[1]
            # # rr, cc = line(math.floor(b + m * 0), 0, math.floor(b + m * 224), 224)
            # rr, cc = line(-math.floor(b/m),0, math.floor((223-b)/m), 223)
            # # cc = cc[np.where(rr < 224)]
            # # cc = cc[np.where(rr >= 0)]
            # rr = rr[np.where(rr < 224)]
            # rr = rr[np.where(rr >= 0)]

            frame[mask] = 255
            try:
                # frame[rr, cc] = 255
                # resized = cv2.resize(frame, (700, 700), interpolation=cv2.INTER_AREA)
                cv2.imshow('video', frame)
            except:
                cv2.imshow('video', frame)
            # resized = cv2.resize(frame, (1920,1080), interpolation=cv2.INTER_AREA)
            # plt.imshow(frame)
            # plt.show()
            # plt.close()
            # cv2.imshow('frame', frame)
            # cv2.waitKey(0)
            # cv2.destroyWindow('frame')
        #     edge_features = np.append(np.reshape(grad, (108, 192, 1)), np.reshape(angle, (108, 192, 1)), axis=2)
        #     frame = cv2.resize(frame_org, (192, 108))
        #     frame = np.reshape(np.append(frame, edge_features, axis=2), (5, 108, 192))
        #     params = model(torch.tensor(frame / 255).unsqueeze(0).to(device, dtype=torch.float))
        #     params[2] = (2 * params[2]) - 1
        #     queue.append(params[0][0, 0])
        #     queue1.append(params[1])
        #     params[0][0, 0] = sum(queue) / len(queue)
        #     params[1] = sum(queue1) / len(queue1)
        #     canvas = draw_h(frame_org, params)

        else:

            video_capture.release()
            cv2.destroyWindow('video')
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyWindow('video')
