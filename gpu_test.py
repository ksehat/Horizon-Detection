import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')
print("Version of Tensorflow: ", tf.__version__)
print("Cuda Availability: ", tf.test.is_built_with_cuda())
print("GPU  Availability: ", tf.test.is_gpu_available())
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
