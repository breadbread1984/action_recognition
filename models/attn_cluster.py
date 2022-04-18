#!/usr/bin/python3

import numpy as np;
import tensorflow as tf;

class Normalize(tf.keras.layers.Layer):
  def __init__(self, **kwargs):
    super(Normalize, self).__init__(**kwargs);
  def build(self, input_shape):
    self.w = self.add_weight(shape = (1,input_shape[-2],1), dtype = tf.float32, trainable = True, initializer = tf.keras.initializers.RandomNormal(mean = 0., stddev = np.sqrt(1./np.sqrt(input_shape[-2]))), name = 'w');
    self.b = self.add_weight(shape = (1,input_shape[-2],1), dtype = tf.float32, trainable = True, initializer = tf.keras.initializers.RandomNormal(mean = 0., stddev = np.sqrt(1./np.sqrt(input_shape[-2]))), name = 'b');
  def call(self, inputs):
    results = self.w * inputs + self.b;
    return results;
  def get_config(self):
    config = super(Normalize, self).get_config();
    return config;
  @classmethod
  def from_config(cls, config):
    return cls(**config);

def ShiftAttention(input_dim, n_att = 32):
  inputs = tf.keras.Input((None, input_dim)); # inputs.shape = (batch, length, channels)
  results = tf.keras.layers.Dense(n_att,
                                   kernel_initializer = tf.keras.initializers.VarianceScaling(mode = 'fan_in', distribution = 'truncated_normal'),
                                   bias_initializer = tf.keras.initializers.VarianceScaling(mode = 'fan_in', distribution = 'normal'))(inputs); # results.shape = (batch, length, n_att)
  attention = tf.keras.layers.Softmax(axis = -2)(results); # attention.shape = (batch, length, n_att)
  weighted = tf.keras.layers.Lambda(lambda x: tf.linalg.matmul(x[0], x[1], transpose_a = True))([attention, inputs]); # weighted.shape = (batch, n_att, channels)
  weighted = Normalize()(weighted); # weighted.shape = (batch, n_att, channels)
  normalized = tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis = -1))(weighted); # normalized.shape = (batch, n_att, channels)
  normalized = tf.keras.layers.Lambda(lambda x, n: x / tf.math.sqrt(n), arguments = {'n': float(n_att)})(normalized); # normalized.shape = (batch, n_att, channels)
  results = tf.keras.layers.Flatten()(normalized); # results.shape = (batch, n_att * channels)
  return tf.keras.Model(inputs = inputs, outputs = results);

def AttentionCluster(vocab_size, seg_num = 100, feature_dims = [1024, 128], cluster_nums = [32, 32], drop_rate = 0.5):
  inputs = [tf.keras.Input((seg_num, dim)) for dim in feature_dims];
  att_outputs = list();
  for input_data, cluster_num in zip(inputs, cluster_nums):
    att_out = ShiftAttention(input_data.shape[-1], cluster_num)(input_data); # att_out.shape = (batch, cluster_num * channels)
    att_outputs.append(att_out);
  out = tf.keras.layers.Concatenate(axis = 1)(att_outputs); # out.shape = (batch, (cluster_num1 + cluster_num2) * channels)
  out = tf.keras.layers.Dropout(rate = drop_rate)(out); # out.shape = (batch, (cluster_num1 + cluster_num2) * channels)
  fc1 = tf.keras.layers.Dense(1024, activation = tf.keras.activations.tanh,
                              kernel_initializer = tf.keras.initializers.VarianceScaling(mode = 'fan_in', distribution = 'truncated_normal'),
                              bias_initializer = tf.keras.initializers.VarianceScaling(mode = 'fan_in', distribution = 'normal'))(out); # fc1.shape = (batch, 1024)
  fc2 = tf.keras.layers.Dense(4096, activation = tf.keras.activations.tanh,
                              kernel_initializer = tf.keras.initializers.VarianceScaling(mode = 'fan_in', distribution = 'truncated_normal'),
                              bias_initializer = tf.keras.initializers.VarianceScaling(mode = 'fan_in', distribution = 'normal'))(fc1); # fc2.shape = (batch, 1024)
  logits = tf.keras.layers.Dense(vocab_size, activation = tf.keras.activations.sigmoid,
                                 kernel_initializer = tf.keras.initializers.VarianceScaling(mode = 'fan_in', distribution = 'truncated_normal'),
                                 bias_initializer = tf.keras.initializers.VarianceScaling(mode = 'fan_in', distribution = 'normal'))(fc2); # logits.shape = (batch, vocab_size)
  return tf.keras.Model(inputs = inputs, outputs = logits);

if __name__ == "__main__":
  atten_cluster = AttentionCluster(100);
  input1 = np.random.normal(size = (4, 100, 1024)); # rgb
  input2 = np.random.normal(size = (4, 100, 128)); # audio
  outputs = atten_cluster([input1, input2]);
  print(outputs.shape);
  tf.keras.utils.plot_model(atten_cluster, to_file = 'atten_cluster.png', show_shapes = True, rankdir = 'TB', expand_nested = True);
