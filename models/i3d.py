#!/usr/bin/python3

import tensorflow as tf;

def Unit3D(in_channels, out_channels, kernel_size = (1,1,1), strides = (1,1,1), activation = 'relu', use_batch_norm = True, use_bias = False):
  inputs = tf.keras.Input((None,None,None,in_channels));
  results = tf.keras.layers.Conv3D(out_channel, kernel_size = kernel_size, strides = strides, padding = 'same', use_bias = use_bias)(inputs);
  results = tf.keras.layers.BatchNormalization()(results);
  results = tf.keras.layers.Activation(activation)(results);
  return tf.keras.Model(inputs = inputs, outputs = results);

def I3D():
  inputs = tf.keras.Input((None,224,224,3));
  results = Unit3D(3, 64, kernel_size = (7,7,7), strides = (2,2,2))(inputs);
  results = tf.keras.layers.MaxPool3D(pool_size = (1,3,3), strides = (1,2,2), padding = 'same')(results);
  results = Unit3D(64, 64, kernel_size = (1,1,1))(results);
  results = Unit3D(192, 64, kernel_size = (3,3,3))(results);
  results = tf.keras.layers.MaxPool3D(pool_size = (1,3,3), strides = (1,2,2), padding = 'same')(results);
  # inception
  branch_0 = Unit3D(64, kernel_size = (1,1,1))(results);
  branch_1 = Unit3D(96, kernel_size = (1,1,1))(results);
  branch_1 = Unit3D(128, kernel_size = (3,3,3))(branch_1);
  branch_2 = Unit3D(16, kernel_size = (1,1,1))(results);
  branch_2 = Unit3D(32, kernel_size = (3,3,3))(branch_2);
  branch_3 = tf.keras.layers.MaxPool3D(pool_size = (3,3,3), strides = (1,1,1), padding = 'same')(results);
  branch_3 = Unit3D(32, kernel_size = (1,1,1))(branch_3);
  results = tf.keras.layers.Concatenate(axis = -1)([branch_0, branch_1, branch_2, branch_3]);
  # inception
  branch_0 = Unit3D(128, kernel_size = (1,1,1))(results);
  branch_1 = Unit3D(128, kernel_size = (1,1,1))(results);
  branch_1 = Unit3D(192, kernel_size = (3,3,3))(branch_1);
  branch_2 = Unit3D(32, kernel_size = (1,1,1))(results);
  branch_2 = Unit3D(96, kernel_size = (3,3,3))(branch_2);
  branch_3 = tf.keras.layers.MaxPool3D(pool_size = (3,3,3), strides = (1,1,1), padding = 'same')(results);
  branch_3 = Unit3D(64, kernel_size = (1,1,1))(branch_3);
  results = tf.keras.layers.Concatenate(axis = -1)([branch_0, branch_1, branch_2, branch_3]);

  results = tf.keras.layers.MaxPool3D(pool_size = (3,3,3), strides = (2,2,2), padding = 'same')(results);
  # inception
  branch_0 = Unit3D(192, kernel_size = (1,1,1))(results);
  branch_1 = Unit3D(96, kernel_size = (1,1,1))(results);
  branch_1 = Unit3D(208, kernel_size = (3,3,3))(branch_1);
  branch_2 = Unit3D(16, kernel_size = (1,1,1))(results);
  branch_2 = Unit3D(48, kernel_size = (3,3,3))(branch_2);
  branch_3 = tf.keras.layers.MaxPool3D(pool_size = (3,3,3), strides = (1,1,1), padding = 'same')(results);
  branch_3 = Unit3D(64, kernel_size = (1,1,1))(branch_3);
  results = tf.keras.layers.Concatenate(axis = -1)([branch_0, branch_1, branch_2, branch_3]);
  # inception
  branch_0 = Unit3D(160, kernel_size = (1,1,1))(results);
  branch_1 = Unit3D(112, kernel_size = (1,1,1))(results);
  branch_1 = Unit3D(224, kernel_size = (3,3,3))(branch_1);
  branch_2 = Unit3D(24, kernel_size = (1,1,1))(results);
  branch_2 = Unit3D(64, kernel_size = (3,3,3))(branch_2);
  branch_3 = tf.keras.layers.MaxPool3D(pool_size = (3,3,3), strides = (1,1,1), padding = 'same')(results);
  branch_3 = Unit3D(64, kernel_size = (1,1,1))(branch_3);
  results = tf.keras.layers.Concatenate(axis = -1)([branch_0, branch_1, branch_2, branch_3]);
  # inception
  branch_0 = Unit3D(128, kernel_size = (1,1,1))(results);
  branch_1 = Unit3D(128, kernel_size = (1,1,1))(results);
  branch_1 = Unit3D(256, kernel_size = (3,3,3))(branch_1);
  branch_2 = Unit3D(24, kernel_size = (1,1,1))(results);
  branch_2 = Unit3D(64, kernel_size = (3,3,3))(branch_2);
  branch_3 = tf.keras.layers.MaxPool3D(pool_size = (3,3,3), strides = (1,1,1), padding = 'same')(results);
  branch_3 = Unit3D(64, kernel_size = (1,1,1))(branch_3);
  results = tf.keras.layers.Concatenate(axis = -1)([branch_0, branch_1, branch_2, branch_3]);
  # inception
