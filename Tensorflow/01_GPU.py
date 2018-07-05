#encoding:utf-8
'''
GPU加速
'''
# https://colab.research.google.com/notebooks/gpu.ipynb#scrollTo=3IEVK-KFxi5Z
import tensorflow as tf
device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))