#!/usr/bin/python3

import tensorflow as tf;

def TemporalShift(length, channels):
  assert channels % 4 == 0;
  inputs = tf.keras.Input((None, None, channels)); # inputs.shape = (batch * length, h, w, c)
  results = tf.keras.layers.Lambda(lambda x, l: tf.reshape(x, (-1, l, tf.shape(x)[-3], tf.shape(x)[-2], tf.shape(x)[-1])), arguments = {'l': length})(inputs); # results.shape = (batch, length, h, w, c)
  results = tf.keras.layers.Lambda(lambda x: tf.pad(x, [[0,0],[1,1],[0,0],[0,0],[0,0]]))(results); # results.shape = (batch, length + 2, h, w, c)
  slice1 = tf.keras.layers.Lambda(lambda x: x[:,:-2,:,:,:x.shape[-1]//4])(results); # backward: slice1.shape = (batch, length, h, w, c/4)
  slice2 = tf.keras.layers.Lambda(lambda x: x[:,2:,:,:,x.shape[-1]//4:x.shape[-1]//2])(results); # forward: slice2.shape = (batch, length, h, w, c/4)
  slice3 = tf.keras.layers.Lambda(lambda x: x[:,1:-1,:,:,x.shape[-1]//2:])(results); # current: slice3.shape = (batch, length, h, w, c/2)
  results = tf.keras.layers.Concatenate(axis = -1)([slice1, slice2, slice3]); # results.shape = (batch, length, h, w, c)
  results = tf.keras.layers.Lambda(lambda x: tf.reshape(x, (-1, tf.shape(x)[-3], tf.shape(x)[-2], tf.shape(x)[-1])))(results); # results.shape = (batch, length, h, w, c)
  return tf.keras.Model(inputs = inputs, outputs = results);

def BottleNeck(length, in_channels, out_channels, stride = 1):
  inputs = tf.keras.Input((None, None, in_channels)); # inputs.shape = (batch * length, h, w, c)
  # 1) residual branch
  # NOTE: call temporal shift in residual branch first
  shifted = TemporalShift(length, in_channels)(inputs);
  conv0 = tf.keras.layers.Conv2D(out_channels, kernel_size = (1,1), padding = 'same', use_bias = False)(shifted);
  conv0 = tf.keras.layers.BatchNormalization()(conv0);
  conv1 = tf.keras.layers.ReLU()(conv0);
  conv1 = tf.keras.layers.Conv2D(out_channels, kernel_size = (3,3), padding = 'same', strides = (stride, stride), use_bias = False)(conv1);
  conv1 = tf.keras.layers.BatchNormalization()(conv1);
  conv2 = tf.keras.layers.ReLU()(conv1);
  conv2 = tf.keras.layers.Conv2D(out_channels * 4, kernel_size = (1,1), padding = 'same', use_bias = False)(conv2);
  results = tf.keras.layers.BatchNormalization()(conv2);
  # 2) skip branch
  if stride != 1  or in_channels != out_channels:
    skip = tf.keras.layers.Conv2D(out_channels * 4, kernel_size = (1,1), padding = 'same', strides = (stride, stride), use_bias = False)(inputs);
    skip = tf.keras.layers.BatchNormalization()(skip);
    results = tf.keras.layers.Add()([results,skip]);
  results = tf.keras.layers.ReLU()(results);
  return tf.keras.Model(inputs = inputs, outputs = results);

def TemporalShiftModule(class_num, length, layers = 50):
  assert layers in [50, 101, 152];
  depths = {
    50: [3, 4, 6, 3],
    101: [3, 4, 23, 3],
    152: [3, 8, 36, 3],
  };
  filters = [64, 128, 256, 512];
  inputs = tf.keras.Input((length, 224, 224, 3)); # inputs.shape = (batch, length, h, w, c)
  # NOTE: reshape video clip to batch of images
  results = tf.keras.layers.Lambda(lambda x: tf.reshape(x, (-1, tf.shape(x)[-3], tf.shape(x)[-2], tf.shape(x)[-1])))(inputs);
  # 1) resnet
  results = tf.keras.layers.Conv2D(64, kernel_size = (7,7), strides = (2,2), padding = 'same', use_bias = False)(results);
  results = tf.keras.layers.BatchNormalization()(results);
  results = tf.keras.layers.ReLU()(results);
  results = tf.keras.layers.MaxPool2D(pool_size = (3,3), strides = (2,2), padding = 'same')(results);
  for block in range(len(depths[layers])):
    for depth in range(depths[layers][block]):
      results = BottleNeck(length, results.shape[-1], filters[block], stride = 2 if depth == 0 and block != 0 else 1)(results);
  # 2) prediction
  results = tf.keras.layers.AveragePooling2D(pool_size = (7,7))(results); # results.shape = (batch * length, 1, 1, 2048)
  results = tf.keras.layers.Lambda(lambda x, l: tf.reshape(x, (-1, l, tf.shape(x)[-1])), arguments = {'l': length})(results); # results.shape = (batch, length, 2048)
  results = tf.keras.layers.Lambda(lambda x: tf.math.reduce_mean(x, axis = 1))(results); # results.shape = (batch, 2048)
  results = tf.keras.layers.Dropout(rate = 0.5)(results);
  results = tf.keras.layers.Dense(class_num, activation = tf.keras.activations.softmax)(results);
  return tf.keras.Model(inputs = inputs, outputs = results);

if __name__ == "__main__":
  import numpy as np;
  a = np.random.normal(size = (4, 10,224,224,3));
  b = TemporalShiftModule(100, 10, 50)(a);
  print(b.shape);
