#!/usr/bin/python3

import tensorflow as tf;

def TemporalShift(channels):
  assert channels % 4 == 0;
  inputs = tf.keras.Input((None, None, None, channels)); # inputs.shape = (batch, length, h, w, c)
  results = tf.keras.layers.Lambda(lambda x: tf.pad(x, [[0,0],[1,1],[0,0],[0,0],[0,0]]))(inputs); # results.shape = (batch, length + 2, h, w, c)
  slice1 = tf.keras.layers.Lambda(lambda x: x[:,:-2,:,:,:x.shape[-1]//4])(results); # backward: slice1.shape = (batch, length, h, w, c/4)
  slice2 = tf.keras.layers.Lambda(lambda x: x[:,2:,:,:,x.shape[-1]//4:x.shape[-1]//2])(results); # forward: slice2.shape = (batch, length, h, w, c/4)
  slice3 = tf.keras.layers.Lambda(lambda x: x[:,1:-1,:,:,x.shape[-1]//2:])(results); # current: slice3.shape = (batch, length, h, w, c/2)
  results = tf.keras.layers.Concatenate(axis = -1)([slice1, slice2, slice3]); # results.shape = (batch, length, h, w, c)
  return tf.keras.Model(inputs = inputs, outputs = results);

def BottleNeck(in_channels, out_channels, strides = 1):
  inputs = tf.keras.Input((None, None, None, in_channels)); # inputs.shape = (batch, length, h, w, c)
  length = tf.keras.layers.Lambda(lambda x: tf.shape(x)[1])(inputs);
  # 1) residual branch
  shifted = TemporalShift(in_channels)(inputs);
  results = tf.keras.layers.Lambda(lambda x: tf.reshape(x, (-1, tf.shape(x)[-3], tf.shape(x)[-2], tf.shape(x)[-1])))(shifted); # results.shape = (batch * length, h, w, c)
  conv0 = tf.keras.layers.Conv2D(out_channels, kernel_size = (1,1), padding = 'same', use_bias = False)(results);
  conv0 = tf.keras.layers.BatchNormalization()(conv0);
  conv1 = tf.keras.layers.ReLU()(conv0);
  conv1 = tf.keras.layers.Conv2D(out_channels, kernel_size = (3,3), padding = 'same', strides = (stride, stride), use_bias = False)(conv1);
  conv1 = tf.keras.layers.BatchNormalization()(conv1);
  conv2 = tf.keras.layers.ReLU()(conv1);
  conv2 = tf.keras.layers.Conv2D(out_channels * 4, kernel_size = (1,1), padding = 'same', use_bias = False)(conv2);
  conv2 = tf.keras.layers.BatchNormalization()(conv2);
  if stride != 1  or in_channels != out_channels:
    results = tf.keras.layers.Lambda(lambda x: tf.reshape(x, (-1, tf.shape(x)[-3], tf.shape(x)[-2], tf.shape(x)[-1])))(inputs); # results.shape = (batch * length, h, w, c)
    skip = tf.keras.layers.Conv2D(out_channels * 4, kernel_size = (1,1), padding = 'same', strides = (stride, stride), use_bias = False)(results);
    skip = tf.keras.layers.BatchNormalization()(skip);
  results = tf.keras.layers.Add()([conv2,skip]);
  results = tf.keras.layers.ReLU()(results);
  results = tf.keras.layers.Lambda(lambda x: tf.reshape(x[0], (-1, x[1], tf.shape(x)[-3], tf.shape(x)[-2], tf.shape(x)[-1])))([results, length]); # results.shape = (batch, length, h, w, c)
  return tf.keras.Model(inputs = inputs, outouts = results);

def TemporalShiftModule(layers = 50):
  assrt layers in [50, 101, 152];
  depth = {
    50: [3, 4, 6, 3],
    101: [3, 4, 23, 3],
    152: [3, 8, 36, 3],
  };
  channels = [64, 128, 256, 512];
  inputs = tf.keras.Input((None, None, None, 3));
  

if __name__ == "__main__":
  import numpy as np;
  a = np.random.normal(size = (4,10,112,112,256));
  b = temporal_shift(256)(a);
  print(b.shape);
