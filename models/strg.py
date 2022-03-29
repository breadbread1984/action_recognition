#!/usr/bin/python3

import tensorflow as tf;

def GraphConvolution(in_channels, out_channels, use_bias = True):
  features = tf.keras.Input((None, in_channels)); # inputs.shape = (batch, node num, in_channels)
  adj = tf.keras.Input((None, None)); # adj.shape = (batch, node num, node num)
  supports = tf.keras.layers.Dense(out_channels, use_bias = use_bias, kernel_initializer = tf.keras.initializers.VarianceScaling())(features); # results.shape = (batch, node num, out_channels)
  outputs = tf.keras.layers.Lambda(lambda x: tf.linalg.matmul(x[0], x[1]))([adj, supports]); # outputs.shap = (batch, node num, out_channels)
  return tf.keras.Model(inputs = (features,adj), outputs = outputs);

def GCN(channels, hidden_channels, dropout_rate = 0.1):
  inputs = tf.keras.Input((None, channels)); # x.shape = (batch, node num, channels)
  adj = tf.keras.Input((None, None)); # adj.shape = (batch, node_num, node num)
  results = GraphConvolution(channels, hidden_channels)([inputs, adj]); # x.shape = (batch, node num, hidden channels)
  results = tf.keras.layers.ReLU()(results);
  results = tf.keras.layers.Dropout(dropout_rate)(results);
  results = GraphConvolution(hidden_channels, channels)([results, adj]); # x.shape = (batch, node num, channels)
  return tf.keras.Model(inputs = (inputs, adj), outputs = results);

def STRG():
  pass;
  
if __name__ == "__main__":
  gcn = GCN(100,20);
  inputs = tf.random.normal(shape = (4, 10, 100));
  adj = tf.random.uniform(minval = 0, maxval = 1, shape = (10,10), dtype = tf.int32);
  outputs = gcn([inputs,adj]);
  print(outputs.shape);
