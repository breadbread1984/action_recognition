#!/usr/bin/python3

import tensorflow as tf;

def Unit3D(in_channels, out_channels, kernel_size = (1,1,1), strides = (1,1,1), activation = 'relu', use_batch_norm = True, use_bias = False):
  inputs = tf.keras.Input((None,None,None,in_channels));
  results = tf.keras.layers.Conv3D(out_channels, kernel_size = kernel_size, strides = strides, padding = 'same', use_bias = use_bias)(inputs);
  if use_batch_norm:
    results = tf.keras.layers.BatchNormalization()(results);
  if activation:
    results = tf.keras.layers.Activation(activation)(results);
  return tf.keras.Model(inputs = inputs, outputs = results);

def I3D(num_classes, channels = 3, drop_rate = 0.1):
  assert num_classes in [2,3]; # 3 for RGB, 2 for optical flow
  inputs = tf.keras.Input((None,224,224, channels));
  results = Unit3D(inputs.shape[-1], 64, kernel_size = (7,7,7), strides = (2,2,2))(inputs);
  results = tf.keras.layers.MaxPool3D(pool_size = (1,3,3), strides = (1,2,2), padding = 'same')(results);
  results = Unit3D(results.shape[-1], 64, kernel_size = (1,1,1))(results);
  results = Unit3D(results.shape[-1], 192, kernel_size = (3,3,3))(results);
  results = tf.keras.layers.MaxPool3D(pool_size = (1,3,3), strides = (1,2,2), padding = 'same')(results);
  
  # inception
  branch_0 = Unit3D(results.shape[-1], 64, kernel_size = (1,1,1))(results);
  branch_1 = Unit3D(results.shape[-1], 96, kernel_size = (1,1,1))(results);
  branch_1 = Unit3D(branch_1.shape[-1], 128, kernel_size = (3,3,3))(branch_1);
  branch_2 = Unit3D(results.shape[-1], 16, kernel_size = (1,1,1))(results);
  branch_2 = Unit3D(branch_2.shape[-1], 32, kernel_size = (3,3,3))(branch_2);
  branch_3 = tf.keras.layers.MaxPool3D(pool_size = (3,3,3), strides = (1,1,1), padding = 'same')(results);
  branch_3 = Unit3D(branch_3.shape[-1], 32, kernel_size = (1,1,1))(branch_3);
  results = tf.keras.layers.Concatenate(axis = -1)([branch_0, branch_1, branch_2, branch_3]);
  
  # inception
  branch_0 = Unit3D(results.shape[-1], 128, kernel_size = (1,1,1))(results);
  branch_1 = Unit3D(results.shape[-1], 128, kernel_size = (1,1,1))(results);
  branch_1 = Unit3D(branch_1.shape[-1], 192, kernel_size = (3,3,3))(branch_1);
  branch_2 = Unit3D(results.shape[-1], 32, kernel_size = (1,1,1))(results);
  branch_2 = Unit3D(branch_2.shape[-1], 96, kernel_size = (3,3,3))(branch_2);
  branch_3 = tf.keras.layers.MaxPool3D(pool_size = (3,3,3), strides = (1,1,1), padding = 'same')(results);
  branch_3 = Unit3D(branch_3.shape[-1], 64, kernel_size = (1,1,1))(branch_3);
  results = tf.keras.layers.Concatenate(axis = -1)([branch_0, branch_1, branch_2, branch_3]);

  results = tf.keras.layers.MaxPool3D(pool_size = (3,3,3), strides = (2,2,2), padding = 'same')(results);
  
  # inception
  branch_0 = Unit3D(results.shape[-1], 192, kernel_size = (1,1,1))(results);
  branch_1 = Unit3D(results.shape[-1], 96, kernel_size = (1,1,1))(results);
  branch_1 = Unit3D(branch_1.shape[-1], 208, kernel_size = (3,3,3))(branch_1);
  branch_2 = Unit3D(results.shape[-1], 16, kernel_size = (1,1,1))(results);
  branch_2 = Unit3D(branch_2.shape[-1], 48, kernel_size = (3,3,3))(branch_2);
  branch_3 = tf.keras.layers.MaxPool3D(pool_size = (3,3,3), strides = (1,1,1), padding = 'same')(results);
  branch_3 = Unit3D(branch_3.shape[-1], 64, kernel_size = (1,1,1))(branch_3);
  results = tf.keras.layers.Concatenate(axis = -1)([branch_0, branch_1, branch_2, branch_3]);
  
  # inception
  branch_0 = Unit3D(results.shape[-1], 160, kernel_size = (1,1,1))(results);
  branch_1 = Unit3D(results.shape[-1], 112, kernel_size = (1,1,1))(results);
  branch_1 = Unit3D(branch_1.shape[-1], 224, kernel_size = (3,3,3))(branch_1);
  branch_2 = Unit3D(results.shape[-1], 24, kernel_size = (1,1,1))(results);
  branch_2 = Unit3D(branch_2.shape[-1], 64, kernel_size = (3,3,3))(branch_2);
  branch_3 = tf.keras.layers.MaxPool3D(pool_size = (3,3,3), strides = (1,1,1), padding = 'same')(results);
  branch_3 = Unit3D(branch_3.shape[-1], 64, kernel_size = (1,1,1))(branch_3);
  results = tf.keras.layers.Concatenate(axis = -1)([branch_0, branch_1, branch_2, branch_3]);
  
  # inception
  branch_0 = Unit3D(results.shape[-1], 128, kernel_size = (1,1,1))(results);
  branch_1 = Unit3D(results.shape[-1], 128, kernel_size = (1,1,1))(results);
  branch_1 = Unit3D(branch_1.shape[-1], 256, kernel_size = (3,3,3))(branch_1);
  branch_2 = Unit3D(results.shape[-1], 24, kernel_size = (1,1,1))(results);
  branch_2 = Unit3D(branch_2.shape[-1], 64, kernel_size = (3,3,3))(branch_2);
  branch_3 = tf.keras.layers.MaxPool3D(pool_size = (3,3,3), strides = (1,1,1), padding = 'same')(results);
  branch_3 = Unit3D(branch_3.shape[-1], 64, kernel_size = (1,1,1))(branch_3);
  results = tf.keras.layers.Concatenate(axis = -1)([branch_0, branch_1, branch_2, branch_3]);
  
  # inception
  branch_0 = Unit3D(results.shape[-1], 112, kernel_size = (1,1,1))(results);
  branch_1 = Unit3D(results.shape[-1],144, kernel_size = (1,1,1))(results);
  branch_1 = Unit3D(branch_1.shape[-1], 288, kernel_size = (3,3,3))(branch_1);
  branch_2 = Unit3D(results.shape[-1], 32, kernel_size = (1,1,1))(results);
  branch_2 = Unit3D(branch_2.shape[-1], 64, kernel_size = (3,3,3))(branch_2);
  branch_3 = tf.keras.layers.MaxPool3D(pool_size = (3,3,3), strides = (1,1,1), padding = 'same')(results);
  branch_3 = Unit3D(branch_3.shape[-1], 64, kernel_size = (1,1,1))(branch_3);
  results = tf.keras.layers.Concatenate(axis = -1)([branch_0, branch_1, branch_2, branch_3]);
  
  # inception
  branch_0 = Unit3D(results.shape[-1], 256, kernel_size = (1,1,1))(results);
  branch_1 = Unit3D(results.shape[-1], 160, kernel_size = (1,1,1))(results);
  branch_1 = Unit3D(branch_1.shape[-1], 320, kernel_size = (3,3,3))(branch_1);
  branch_2 = Unit3D(results.shape[-1], 32, kernel_size = (1,1,1))(results);
  branch_2 = Unit3D(branch_2.shape[-1], 128, kernel_size = (3,3,3))(branch_2);
  branch_3 = tf.keras.layers.MaxPool3D(pool_size = (3,3,3), strides = (1,1,1), padding = 'same')(results);
  branch_3 = Unit3D(branch_3.shape[-1], 128, kernel_size = (1,1,1))(branch_3);
  results = tf.keras.layers.Concatenate(axis = -1)([branch_0, branch_1, branch_2, branch_3]);
  
  results = tf.keras.layers.MaxPool3D(pool_size = (2,2,2), strides = (2,2,2), padding = 'same')(results);
  
  # inception
  branch_0 = Unit3D(results.shape[-1], 256, kernel_size = (1,1,1))(results);
  branch_1 = Unit3D(results.shape[-1], 160, kernel_size = (1,1,1))(results);
  branch_1 = Unit3D(branch_1.shape[-1], 320, kernel_size = (3,3,3))(branch_1);
  branch_2 = Unit3D(results.shape[-1], 32, kernel_size = (1,1,1))(results);
  branch_2 = Unit3D(branch_2.shape[-1], 128, kernel_size = (3,3,3))(branch_2);
  branch_3 = tf.keras.layers.MaxPool3D(pool_size = (3,3,3), strides = (1,1,1), padding = 'same')(results);
  branch_3 = Unit3D(branch_3.shape[-1], 128, kernel_size = (1,1,1))(branch_3);
  results = tf.keras.layers.Concatenate(axis = -1)([branch_0, branch_1, branch_2, branch_3]);
  
  # inception
  branch_0 = Unit3D(results.shape[-1], 384, kernel_size = (1,1,1))(results);
  branch_1 = Unit3D(results.shape[-1], 192, kernel_size = (1,1,1))(results);
  branch_1 = Unit3D(branch_1.shape[-1], 384, kernel_size = (3,3,3))(branch_1);
  branch_2 = Unit3D(results.shape[-1], 48, kernel_size = (1,1,1))(results);
  branch_2 = Unit3D(branch_2.shape[-1], 128, kernel_size = (3,3,3))(branch_2);
  branch_3 = tf.keras.layers.MaxPool3D(pool_size = (3,3,3), strides = (1,1,1), padding = 'same')(results);
  branch_3 = Unit3D(branch_3.shape[-1], 128, kernel_size = (3,3,3))(branch_3);
  results = tf.keras.layers.Concatenate(axis = -1)([branch_0, branch_1, branch_2, branch_3]);
  
  results = tf.keras.layers.AveragePooling3D(pool_size = (2,7,7), strides = (1,1,1), padding = 'valid')(results);
  results = tf.keras.layers.Dropout(drop_rate)(results);
  logits = Unit3D(results.shape[-1], num_classes, kernel_size = (1,1,1), activation = None, use_batch_norm = False, use_bias = True)(results);
  avg_logits = tf.keras.layers.Lambda(lambda x: tf.math.reduce_mean(tf.squeeze(x, axis = (2,3)), axis = 1))(logits); # avg_logits.shape = (batch, 100)
  return tf.keras.Model(inputs = inputs, outputs = avg_logits);

if __name__ == "__main__":
  inputs = tf.random.normal(shape = (4,64,224,224,3));
  outputs = I3D(100)(inputs);
  print(outputs.shape);
