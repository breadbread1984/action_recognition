#!/usr/bin/python3

import numpy as np;
import tensorflow as tf;

def GCN(in_channel, out_channel, spatial_kernel_size, spatial_num = 18):
  inputs = tf.keras.Input((None, spatial_num, in_channel)); # inputs.shape = (batch, temporal_num, spatial_num, in_channel)
  A = tf.keras.Input((spatial_num, spatial_num), batch_size = spatial_kernel_size); # A.shape = (spatial_kernel_size, spatial_num, spatial_num)
  results = tf.keras.layers.Dense(spatial_kernel_size * out_channel)(inputs); # results.shape = (batch, temporal_num, spatial_num, spatial_kernel_size * out_channel)
  results = tf.keras.layers.Lambda(lambda x, k, c: tf.transpose(tf.reshape(x, (tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], k, c)), (0,1,3,2,4)), arguments = {'k': spatial_kernel_size, 'c': out_channel})(results); # results.shape = (batch, temporal_num, spatial_kernel_size, spatial_num, out_channel)
  results = tf.keras.layers.Lambda(lambda x: tf.reshape(x, (tf.shape(x)[0], tf.shape(x)[1], -1, tf.shape(x)[4])))(results); # results.shape = (batch, temporal_num, spatial_kernel_size * spatial_num, out_channel)
  a = tf.keras.layers.Lambda(lambda x: tf.reshape(x, (-1, tf.shape(x)[-1])))(A); # a.shape = (spatial_kernel_size * spatial_num, spatial_num)
  results = tf.keras.layers.Lambda(lambda x: tf.transpose(tf.linalg.matmul(x[0], x[1], transpose_a = True), (0,1,3,2)))([results, a]); # results.shape = (batch, temporal_num, spatial_num, out_channel)
  return tf.keras.Model(inputs = (inputs, A), outputs = results);

def TCN(channel, stride = 1, temporal_kernel_size = 9, drop_rate = 0):
  # temporal convolutional network uses conv 1x1 not graph convolution
  inputs = tf.keras.Input((None, None, channel)); # inputs.shape = (batch, temporal_num, spatial_num, channel)
  results = tf.keras.layers.BatchNormalization()(inputs);
  results = tf.keras.layers.ReLU()(results);
  results = tf.keras.layers.Conv2D(channel, kernel_size = (temporal_kernel_size, 1), strides = (stride, 1), padding = 'same')(results); # results.shape = (batch, temporal_num, spatial_num, channel)
  results = tf.keras.layers.BatchNormalization()(results);
  results = tf.keras.layers.Dropout(rate = 0)(results);
  return tf.keras.Model(inputs = inputs, outputs = results);

def STGCNLayer(in_channel, out_channel, spatial_kernel_size, temporal_kernel_size = 9, stride = 1, use_skip = True, spatial_num = 18, drop_rate = 0):
  inputs = tf.keras.Input((None, spatial_num, in_channel)); # inputs.shape = (batch, temporal_num, spatial_num, in_channel)
  A = tf.keras.Input((spatial_num, spatial_num), batch_size = spatial_kernel_size);
  # 1) residual branch
  residual = GCN(in_channel, out_channel, spatial_kernel_size, spatial_num)([inputs, A]);
  residual = TCN(out_channel, stride, temporal_kernel_size, drop_rate)(residual);
  # 2) skip branch
  if use_skip and stride == 1 and in_channel == out_channel:
    skip = inputs;
  elif use_skip:
    skip = tf.keras.layers.Conv2D(out_channel, kernel_size = (1,1), strides = (stride, 1), padding = 'same')(inputs);
    skip = tf.keras.layers.BatchNormalization()(skip);
  else:
    skip = tf.keras.layers.Lambda(lambda x: tf.zeros_like(x))(residual);
  # 3) output
  results = tf.keras.layers.Add()([residual, skip]);
  results = tf.keras.layers.ReLU()(results);
  return tf.keras.Model(inputs = (inputs, A), outputs = results);

class Weighted(tf.keras.layers.Layer):
  def __init__(self, **kwargs):
    super(Weighted, self).__init__(**kwargs);
  def build(self, input_shape):
    self.importance = self.add_weight(shape = (input_shape[0],), dtype = tf.float32, trainable = True, initializer = tf.keras.initializers.Ones(), name = 'importance');
  def call(self, inputs):
    # inputs.shape = (spatial_kernel_size, spatial_num, spatial_num)
    results = inputs * tf.reshape(self.importance, (-1, 1, 1));
    return results;
  def get_config(self):
    config = super(Weighted, self).get_config();
    return config;
  @classmethod
  def from_config(cls, config):
    return cls(**config);

def STGCN(num_class, channel, spatial_kernel_size, temporal_kernel_size = 9, edge_weighting = True, spatial_num = 18, temporal_num = 300):
  inputs = tf.keras.Input((None, temporal_num, spatial_num, channel)); # inputs.shape = (batch, instance_num, temporal_num, spatial_num, channel)
  A = tf.keras.Input((spatial_num, spatial_num), batch_size = spatial_kernel_size); # A.shape = (spatial_kernel_size, spatial_num, spatial_num)
  results = tf.keras.layers.Lambda(lambda x: tf.reshape(x, (tf.shape(x)[0] * tf.shape(x)[1], x.shape[2], x.shape[3] * x.shape[4])))(inputs); # results.shape = (batch * instance_num, temporal_num, spatial_num * channel)
  results = tf.keras.layers.BatchNormalization()(results);
  results = tf.keras.layers.Lambda(lambda x, s: tf.reshape(x, (tf.shape(x)[0], tf.shape(x)[1], s, -1)), arguments = {'s': spatial_num})(results); # results.shape = (batch * instance_num, temporal_num, spatial_num, channel)
  weighted_adj = Weighted()(A) if edge_weighting else A;
  results = STGCNLayer(channel, 64, spatial_kernel_size, temporal_kernel_size, 1)([results, weighted_adj]);
  weighted_adj = Weighted()(A) if edge_weighting else A;
  results = STGCNLayer(64, 64, spatial_kernel_size, temporal_kernel_size, 1)([results, weighted_adj]);
  weighted_adj = Weighted()(A) if edge_weighting else A;
  results = STGCNLayer(64, 64, spatial_kernel_size, temporal_kernel_size, 1)([results, weighted_adj]);
  weighted_adj = Weighted()(A) if edge_weighting else A;
  results = STGCNLayer(64, 64, spatial_kernel_size, temporal_kernel_size, 1)([results, weighted_adj]);
  weighted_adj = Weighted()(A) if edge_weighting else A;
  results = STGCNLayer(64, 128, spatial_kernel_size, temporal_kernel_size, 2)([results, weighted_adj]);
  weighted_adj = Weighted()(A) if edge_weighting else A;
  results = STGCNLayer(128, 128, spatial_kernel_size, temporal_kernel_size, 1)([results, weighted_adj]);
  weighted_adj = Weighted()(A) if edge_weighting else A;
  results = STGCNLayer(128, 128, spatial_kernel_size, temporal_kernel_size, 1)([results, weighted_adj]);
  weighted_adj = Weighted()(A) if edge_weighting else A;
  results = STGCNLayer(128, 256, spatial_kernel_size, temporal_kernel_size, 2)([results, weighted_adj]);
  weighted_adj = Weighted()(A) if edge_weighting else A;
  results = STGCNLayer(256, 256, spatial_kernel_size, temporal_kernel_size, 1)([results, weighted_adj]);
  weighted_adj = Weighted()(A) if edge_weighting else A;
  results = STGCNLayer(256, 256, spatial_kernel_size, temporal_kernel_size, 1)([results, weighted_adj]);
  results = tf.keras.layers.Lambda(lambda x: tf.math.reduce_mean(x, axis = [-2, -3]))(results); # results.shape = (batch * instance_num, 256)
  results = tf.keras.layers.Lambda(lambda x: tf.reshape(x[0], (tf.shape(x[1])[0], -1, 256)))([results, inputs]); # results.shape = (batch, instance_num, 256)
  results = tf.keras.layers.Lambda(lambda x: tf.math.reduce_mean(x, axis = 1))(results); # results.shape = (batch, 256)
  results = tf.keras.layers.Dense(num_class, activation = tf.keras.activations.softmax)(results); # results.shape = (batch, num_class)
  return tf.keras.Model(inputs = (inputs, A), outputs = results);

def get_adjacent(keypoint_method = 'openpose', adj_method = 'uniform', max_hop = 1, dilation = 1):
  assert keypoint_method in ['openpose', 'ntu-rgb+d', 'nut_edge'];
  assert adj_method in ['uniform', 'distance', 'spatial'];
  # 1) get edge matrix
  if keypoint_method == 'openpose':
    node_num = 18;
    adj = np.eye(node_num);
    adj[4,3] = adj[3,2] = adj[7,6] = adj[6,5] = adj[13,12] = adj[12,11] = \
    adj[10,9] = adj[9,8] = adj[11,5] = adj[8,2] = adj[5,1] = adj[2,1] = \
    adj[0,1] = adj[15,0] = adj[14,0] = adj[17,15] = adj[16,14] = 1;
    adj = np.maximum(np.transpose(adj), adj);
    center = 1;
  elif keypoint_method == 'ntu-rgb+d':
    node_num = 25;
    adj = np.eye(node_num);
    adj[0,1] = adj[1,20] = adj[2,20] = adj[3,2] = adj[4,20] = \
    adj[5,4] = adj[6,5] = adj[7,6] = adj[8,20] = adj[9,8] = \
    adj[10,9] = adj[11,10] = adj[12,0] = adj[13,12] = adj[14,13] = \
    adj[15,14] = adj[16,0] = adj[17,16] = adj[18,17] = adj[19,18] = \
    adj[21,22] = adj[22,7] = adj[23,24] = adj[24,11] = 1;
    adj = np.max(np.transpose(adj), adj);
    center = 21 - 1;
  elif keypoint_method == 'ntu_edge':
    node_num = 24;
    adj = np.eye(node_num);
    adj[0,1] = adj[2,1] = adj[3,2] = adj[4,1] = adj[5,4] = adj[6,5] = \
    adj[7,6] = adj[8,1] = adj[9,8] = adj[10,9] = adj[11,10] = \
    adj[12,0] = adj[13,12] = adj[14,13] = adj[15,14] = adj[16,0] = \
    adj[17,16] = adj[18,17] = adj[19,18] = adj[20,21] = adj[21,7] = \
    adj[22,23] = adj[23, 11] = 1;
    adj = np.max(np.transpose(adj), adj);
    center = 2;
  else:
    raise Exception('unknown method!');
  # 2) normalized_digraph
  D1 = np.sum(adj, 0); # in degree
  Dn = np.zeros_like(adj);
  for i, d1 in enumerate(D1):
    if d1 > 0: Dn[i,i] = d1**(-1);
  normalized_adjacency = np.dot(adj, Dn);
  hop_dis = np.zeros_like(adj) + np.inf;
  transfer_mat = [np.linalg.matrix_power(adj, d) for d in range(max_hop + 1)];
  arrive_mat = np.stack(transfer_mat) > 0;
  # NOTE: assign distance reversedly to assign distance as low as possible.
  for d in range(max_hop, -1, -1):
    hop_dis[arrive_mat[d]] = d;
  if adj_method == 'uniform':
    A = np.expand_dims(normalized_adjacency, 0); # A.shape = (1, node_num, node_num)
    spatial_kernel_size = 1;
  elif adj_method == 'distance':
    # separate adjacent matrix to multiple matrix according hops number
    A = np.zeros((len(valid_hop), node_num, node_num));
    for i, hop in enumerate(range(0, max_hop + 1, dilation)):
      A[i][hop_dis == hop] = normalized_adjacency[hop_dis == hop];
  elif adj_method == 'spatial':
    A = list();
    for hop in range(0, max_hop + 1, dilation):
      a_root = np.zeros((node_num, node_num));
      a_close = np.zeros((node_num, node_num));
      a_further = np.zeros((node_num, node_num));
      for i in range(node_num):
        for j in range(node_num):
          if hop_dis[j, i] == hop:
            if hop_dis[j, center] == hop_dis[i, center]:
              a_root[j, i] = normalized_adjacency[j, i];
            elif hop_dis[j, center] > hop_dis[i, center]:
              a_close[j, i] = normalized_adjacency[j, i];
            else:
              a_further[j, i] = normalized_adjacency[j, i];
      if hop == 0:
        A.append(a_root);
      else:
        A.append(a_root + a_close);
        A.append(a_further);
    A = np.stack(A);
  else:
    raise Exception('unknown method!');
  return A, node_num;

if __name__ == "__main__":
  A, node_num = get_adjacent('openpose', 'spatial');
  print(A, node_num);
  stgcn = STGCN(10, 2048, A.shape[0], spatial_num = node_num);
  inputs = tf.random.normal(shape = (4, 2, 300, node_num, 2048));
  outputs = stgcn([inputs,A]);
  print(outputs.shape);
