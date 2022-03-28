#!/usr/bin/python3

import tensorflow as tf;

def C3D(class_num):
  inputs = tf.keras.Input((16,112,112,3)); # inputs.shape = (batch,16,112,112,3)
  results = tf.keras.layers.Conv3D(64, kernel_size = (3,3,3), padding = 'same')(inputs); # results.shape = (batch,16,112,112,64)
  results = tf.keras.layers.MaxPool3D(pool_size = (1,2,2), strides = (1,2,2), padding = 'same')(results); # results.shape = (batch,16,56,56,64)
  results = tf.keras.layers.Conv3D(128, kernel_size = (3,3,3), padding = 'same')(results); # results.shape = (batch,16,56,56,128)
  results = tf.keras.layers.MaxPool3D(pool_size = (2,2,2), strides = (2,2,2), padding = 'same')(results); # results.shape = (batch,8,28,28,128)
  results = tf.keras.layers.Conv3D(256, kernel_size = (3,3,3), padding = 'same')(results); # results.shape = (batch,8,28,28,128)
  results = tf.keras.layers.Conv3D(256, kernel_size = (3,3,3), padding = 'same')(results); # results.shape = (batch,8,28,28,128)
  results = tf.keras.layers.MaxPool3D(pool_size = (2,2,2), strides = (2,2,2), padding = 'same')(results); # results.shape = (batch,4,14,14,256)
  results = tf.keras.layers.Conv3D(512, kernel_size = (3,3,3), padding = 'same')(results); # results.shape = (batch,4,14,14,256)
  results = tf.keras.layers.Conv3D(512, kernel_size = (3,3,3), padding = 'same')(results); # results.shape = (batch,4,14,14,256)
  results = tf.keras.layers.MaxPool3D(pool_size = (2,2,2), strides = (2,2,2), padding = 'same')(results); # results.shape = (batch,2,7,7,512)
  results = tf.keras.layers.Conv3D(512, kernel_size = (3,3,3), padding = 'same')(results); # results.shape = (batch,2,7,7,512)
  results = tf.keras.layers.Conv3D(512, kernel_size = (3,3,3), padding = 'same')(results); # results.shape = (batch,2,7,7,512)
  results = tf.keras.layers.MaxPool3D(pool_size = (2,2,2), strides = (2,2,2), padding = 'same')(results); # rsults.shape = (batch,1,3,3,512)
  results = tf.keras.layers.Flatten()(results); # results.shape = (batch, 9*512)
  results = tf.keras.layers.Dropout(0.1)(results); # results.shape = (batch, 9*512)
  results = tf.keras.layers.Dense(units = 4096)(results); # results.shape = (batch, 4096)
  results = tf.keras.layers.Dropout(0.1)(results); # results.shape = (batch 4096)
  results = tf.keras.layers.Dense(units = 4096)(results); # results.shape = (batch, 4096)
  results = tf.keras.layers.Dropout(0.1)(results); # results.shape = (batch, 4096)
  results = tf.keras.layers.Dense(units = class_num)(results); # results.shape = (batch, class_num)
  return tf.keras.Model(inputs = inputs, outputs = results);

if __name__ == "__main__":
  model = C3D(100);
  model.save('c3d.h5');
