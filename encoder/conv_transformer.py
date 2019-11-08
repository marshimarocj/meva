import json
import cPickle
import random
import enum
import sys
import os
sys.path.append('../')

import tensorflow as tf
import numpy as np

import framework.model.module


class Config(framework.model.module.ModuleConfig):
  def __init__(self):
    framework.model.module.ModuleConfig.__init__(self)

    self.num_head = 8

    self.dim_query_embed = 512
    self.dim_key_embed = 512
    self.dim_val_embed = 512

    self.dim_input = 512

    self.dim_hidden = 2048
    self.dim_output = 512
    self.base = 10000.

    self.num_step = 10
    self.kernel_size = 16
    self.dropout = .1
    self.learn_pe = False

  def _assert(self):
    assert self.dim_output == self.dim_input
    assert self.dim_query_embed == self.dim_key_embed
    assert self.dim_query_embed % self.num_head == 0
    assert self.dim_val_embed % self.num_head == 0
    assert self.num_step >= self.kernel_size
    assert self.kernel_size % 2 == 1


class Encoder(framework.model.module.AbstractModule):
  name_scope = 'encoder.conv_transformer'

  class InKey(enum.Enum):
    FT = 'ft' # (None, num_step, dim_ft)
    MASK = 'mask' # (None, num_step)
    IS_TRN = 'is_training'

  class OutKey(enum.Enum):
    OUTPUT = 'output'

  def _set_submods(self):
    return {}

  def _build_parameter_graph(self):
    with tf.variable_scope(self.name_scope):
      self.query_W = tf.contrib.framework.model_variable('query_W',
        shape=(1, self._config.dim_input, self._config.dim_query_embed), dtype=tf.float32,
        initializer=tf.contrib.layers.xavier_initializer())
      self.key_W = tf.contrib.framework.model_variable('key_W',
        shape=(1, self._config.dim_input, self._config.dim_key_embed), dtype=tf.float32,
        initializer=tf.contrib.layers.xavier_initializer())
      self.value_W = tf.contrib.framework.model_variable('value_W',
        shape=(1, self._config.dim_input, self._config.dim_val_embed), dtype=tf.float32,
        initializer=tf.contrib.layers.xavier_initializer())
      self.att_W = tf.contrib.framework.model_variable('att_W',
        shape=(1, self._config.dim_val_embed, self._config.dim_output), dtype=tf.float32,
        initializer=tf.contrib.layers.xavier_initializer())
      self._weights.append(self.query_W)
      self._weights.append(self.key_W)
      self._weights.append(self.value_W)
      self._weights.append(self.att_W)

      self.fc_Ws = []
      self.fc_Bs = []
      dim_inputs = [self._config.dim_val_embed, self._config.dim_hidden]
      dim_outputs = [self._config.dim_hidden, self._config.dim_output]
      for i in range(2):
        fc_W = tf.contrib.framework.model_variable('fc_W_%d'%i,
          shape=(1, dim_inputs[i], dim_outputs[i]), dtype=tf.float32,
          initializer=tf.contrib.layers.xavier_initializer())
        fc_B = tf.contrib.framework.model_variable('fc_B_%d'%i,
          shape=(dim_outputs[i],), dtype=tf.float32,
          initializer=tf.constant_initializer(0.))
        self.fc_Ws.append(fc_W)
        self.fc_Bs.append(fc_B)
        self._weights.append(fc_W)
        self._weights.append(fc_B)

  def get_out_ops_in_mode(self, in_ops, mode, **kwargs):
    with tf.variable_scope(self.name_scope):
      fts = in_ops[self.InKey.FT]
      masks = in_ops[self.InKey.MASK]
      is_trn = in_ops[self.InKey.IS_TRN]
      batch_size = tf.shape(fts)[0]

      kernel_size = self._config.kernel_size
      num_step = self._config.num_step
      dim_input = self._config.dim_input
      dim_key_embed = self._config.dim_key_embed
      dim_val_embed = self._config.dim_val_embed

      # shift convolution
      shift_filter = tf.eye(kernel_size, kernel_size)
      shift_filter = tf.expand_dims(shift_filter, 1) # (K, 1, K)
      fts = tf.transpose(fts, [0, 2, 1])
      fts = tf.reshape(fts, (-1, num_step, 1)) # (None*dim_input, num_step, 1)
      paddings = tf.constant([[0, 0], [kernel_size/2, kernel_size/2], [0, 0]])
      fts = tf.pad(fts, paddings)
      fts_shift = tf.nn.conv1d(fts, shift_filter, 1, 'VALID') # (None*dim_input, num_step, K)
      fts_shift = tf.reshape(fts_shift, (-1, dim_input, num_step, kernel_size))

      masks = tf.expand_dims(masks, 2) # (None, num_step, 1)
      masks = tf.pad(masks, paddings)
      masks_shift = tf.nn.conv1d(masks, shift_filter, 1, 'VALID') # (None, num_step, K)
      # print masks.get_shape(), masks_shift.get_shape()

      # embed, query, key, val
      pos_embeds = gen_relative_position_encoding(kernel_size, dim_input) # (K, dim_input)
      pos_embeds = tf.reshape(tf.transpose(pos_embeds, [1, 0]), (1, dim_input, 1, kernel_size))
      fts_shift += pos_embeds
      fts_shift = tf.transpose(fts_shift, (0, 2, 3, 1)) # (None, num_step, K, dim_input)

      query_embeds = tf.nn.conv1d(fts_shift[:, :, kernel_size/2], self.query_W, 1, 'SAME') # (None, num_step, dim_query)
      query_embeds = tf.expand_dims(query_embeds, 2) # (None, num_step, 1, dim_query)
      fts_shift = tf.reshape(fts_shift, (-1, num_step*kernel_size, dim_input)) # (None, num_step*K, dim_input)
      key_shift_embeds = tf.nn.conv1d(fts_shift, self.key_W, 1, 'VALID') 
      key_shift_embeds = tf.reshape(key_shift_embeds, (-1, num_step, kernel_size, dim_key_embed)) # (None, num_step, K, dim_key_embed)
      val_shift_embeds = tf.nn.conv1d(fts_shift, self.value_W, 1, 'VALID')
      val_shift_embeds = tf.reshape(val_shift_embeds, (-1, num_step, kernel_size, dim_val_embed)) # (None, num_step, K, dim_val_embed)

      # convolutional self-attention sub-layer
      dim_head = self._config.dim_query_embed / self._config.num_head
      dim_head_val = self._config.dim_val_embed / self._config.num_head
      context_vals = []
      for h in range(self._config.num_head):
        alpha = tf.reduce_sum(
          query_embeds[:, :, :, h*dim_head:(h+1)*dim_head] * key_shift_embeds[:, :, :, h*dim_head:(h+1)*dim_head], 3)
        alpha /= dim_head**0.5 # (None, num_step, K)
        alpha = tf.nn.softmax(alpha, 2)
        alpha *= masks_shift
        alpha_sum = tf.reduce_sum(alpha, 2, keepdims=True)
        alpha_sum = tf.where(alpha_sum > 0., alpha_sum, tf.ones_like(alpha_sum))
        alpha /= alpha_sum
        alpha = tf.expand_dims(alpha, 3) # (None, num_step, K, 1)
        head_val_embeds = val_shift_embeds[:, :, :, h*dim_head_val:(h+1)*dim_head_val] # (None, num_step, K, dim_head_val)
        head_context_val = tf.reduce_sum(alpha * head_val_embeds, 2) # (None, num_step, dim_head_val)
        context_vals.append(head_context_val)
      context_vals = tf.concat(context_vals, 2)
      outputs = tf.nn.conv1d(context_vals, self.att_W, 1, 'VALID') # (None, num_step, dim_output)
      outputs = tf.layers.dropout(outputs, rate=self._config.dropout, training=is_trn)

      # add & norm
      outputs += in_ops[self.InKey.FT]
      outputs = tf.contrib.layers.layer_norm(outputs, begin_norm_axis=2)

      # feed forward sub-layer
      inputs = outputs
      for i in range(2):
        outputs = tf.nn.conv1d(outputs, self.fc_Ws[i], 1, 'VALID')
        outputs = tf.nn.bias_add(outputs, self.fc_Bs[i])
        if i == 0:
          outputs = tf.nn.relu(outputs)
      outputs = tf.layers.dropout(outputs, rate=self._config.dropout, training=is_trn)

      # add & norm
      outputs += inputs
      outputs = tf.contrib.layers.layer_norm(outputs, begin_norm_axis=2)

      return {
        self.OutKey.OUTPUT: outputs,
      }


def gen_relative_position_encoding(kernel_size, dim_input, base=10000.):
  pos = tf.to_float(tf.range(-(kernel_size/2), kernel_size/2+1))
  num_timescales = dim_input // 2
  log_timescale_increment = tf.log(base) /(tf.to_float(num_timescales) - 1)
  inv_timescales = tf.exp(tf.to_float(tf.range(num_timescales)) * -log_timescale_increment)
  scaled_time = tf.expand_dims(pos, 1) * tf.expand_dims(inv_timescales, 0)
  signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1) # (K, dim_input)
  return signal
