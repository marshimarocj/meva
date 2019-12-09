import json
import cPickle
import random
import enum
import sys
import os
sys.path.append('../')

import tensorflow as tf
import numpy as np
from sklearn.metrics import average_precision_score

import framework.model.module
import framework.model.trntst
import framework.model.data
import framework.impl.encoder.pca
import op_util
import encoder.conv_transformer
import data
import birnn

PCA = 'pca'
ENC = 'transformer'


class ModelConfig(framework.model.module.ModelConfig):
  def __init__(self):
    framework.model.module.ModelConfig.__init__(self)

    self.subcfgs[ENC] = encoder.conv_transformer.Config()
    self.subcfgs[PCA] = framework.impl.encoder.pca.Config()
    self.num_pos_class = 12
    self.dim_hiddens = []
    self.focal_loss = False
    self.norms = [False]
    self.shift = 4
    self.num_layer = 1

  def _assert(self):
    assert self.subcfgs[ENC].dim_input == self.subcfgs[PCA].dim_output


def gen_cfg(**kwargs):
  cfg = ModelConfig()
  cfg.val_iter = -1
  cfg.monitor_iter = 50
  cfg.trn_batch_size = 32
  cfg.tst_batch_size = 32
  cfg.base_lr = 1e-4
  cfg.num_epoch = 100
  cfg.num_pos_class = kwargs['num_pos_class']
  cfg.dim_hiddens = kwargs['dim_hiddens']
  cfg.shift = kwargs['shift']
  cfg.focal_loss = kwargs['focal_loss']
  cfg.val_loss = False
  cfg.norms = kwargs['norms']
  cfg.num_layer = kwargs['num_layer']

  transformer_cfg = cfg.subcfgs[ENC]
  transformer_cfg.num_head = kwargs['num_head']
  transformer_cfg.dim_query_embed = kwargs['dim_embed']
  transformer_cfg.dim_key_embed = kwargs['dim_embed']
  transformer_cfg.dim_val_embed = kwargs['dim_embed']
  transformer_cfg.dim_input = kwargs['dim_embed']
  transformer_cfg.dim_output = kwargs['dim_embed']
  transformer_cfg.dim_hidden = kwargs['dim_embed']*4
  transformer_cfg.num_step = kwargs['num_step']
  transformer_cfg.kernel_size = kwargs['kernel_size']

  pca_cfg = cfg.subcfgs[PCA]
  pca_cfg.dim_ft = kwargs['dim_ft']
  pca_cfg.dim_output = kwargs['dim_embed']

  return cfg


class Model(framework.model.module.AbstractModel):
  name_scope = 'conv_transformer.Model'

  class InKey(enum.Enum):
    FT = 'fts'
    MASK = 'mask'
    IS_TRN = 'is_training'
    LABEL = 'label'
    LABEL_MASK = 'label_mask'

  class OutKey(enum.Enum):
    LOGIT = 'logit'
    PREDICT = 'predict'

  def _set_submods(self):
    out = {
      PCA: framework.impl.encoder.pca.Encoder1D(self._config.subcfgs[PCA]),
    }
    for i in range(self._config.num_layer):
      out[ENC + '_%d'%i] = encoder.conv_transformer.Encoder(self._config.subcfgs[ENC])
    return out

  def _add_input_in_mode(self, mode):
    pca_cfg = self._config.subcfgs[PCA]
    transformer_cfg = self._config.subcfgs[ENC + '_0']
    with tf.variable_scope(self.name_scope):
      fts = tf.placeholder(
        tf.float32, shape=(None, transformer_cfg.num_step, pca_cfg.dim_ft), name=self.InKey.FT.value)
      masks = tf.placeholder(
        tf.float32, shape=(None, transformer_cfg.num_step), name=self.InKey.MASK.value)
      is_training = tf.placeholder(
        tf.bool, shape=(),name=self.InKey.IS_TRN.value)
      if mode == framework.model.module.Mode.TRN_VAL:
        label = tf.placeholder(
          tf.float32, shape=(None, transformer_cfg.num_step, self._config.num_pos_class), name=self.InKey.LABEL.value)
        label_mask = tf.placeholder(
          tf.float32, shape=(None, transformer_cfg.num_step, self._config.num_pos_class), name=self.InKey.LABEL_MASK.value)
        return {
          self.InKey.FT: fts,
          self.InKey.MASK: masks,
          self.InKey.LABEL: label,
          self.InKey.LABEL_MASK: label_mask,
          self.InKey.IS_TRN: is_training,
        }
      else:
        return {
          self.InKey.FT: fts,
          self.InKey.MASK: masks,
          self.InKey.IS_TRN: is_training,
        }

  def _build_parameter_graph(self):
    transformer_cfg = self._config.subcfgs[ENC + '_0']
    self.fc_Ws = []
    self.fc_Bs = []
    with tf.variable_scope(self.name_scope):
      dim_input = transformer_cfg.dim_output
      dim_inputs = [dim_input] + self._config.dim_hiddens[:-1]
      dim_outputs = self._config.dim_hiddens
      i = 0
      for dim_input, dim_output in zip(dim_inputs, dim_outputs):
        fc_W = tf.contrib.framework.model_variable('fc_W_%d'%i,
          shape=(1, dim_input, dim_output), dtype=tf.float32,
          initializer=tf.contrib.layers.xavier_initializer())
        fc_B = tf.contrib.framework.model_variable('fc_B_%d'%i,
          shape=(dim_output,), dtype=tf.float32,
          initializer=tf.constant_initializer(0.))
        self.fc_Ws.append(fc_W)
        self.fc_Bs.append(fc_B)
        self._weights.append(fc_W)
        self._weights.append(fc_B)
        dim_input = dim_output
        i += 1

      self.sigmoid_W = tf.contrib.framework.model_variable('sigmoid_W',
        shape=(1, self._config.dim_hiddens[-1], self._config.num_pos_class), dtype=tf.float32,
        initializer=tf.contrib.layers.xavier_initializer())
      self.sigmoid_B = tf.contrib.framework.model_variable('sigmoid_B',
        shape=(self._config.num_pos_class,), dtype=tf.float32,
        initializer=tf.constant_initializer(0.))
      self._weights.append(self.sigmoid_W)
      self._weights.append(self.sigmoid_B)

  def get_out_ops_in_mode(self, in_ops, mode):
    out_ops = self.submods[PCA].get_out_ops_in_mode({
      self.submods[PCA].InKey.FT: in_ops[self.InKey.FT],
    }, mode)
    inputs = out_ops[self.submods[PCA].OutKey.EMBED]

    for i in range(self._config.num_layer):
      out_ops = self.submods[ENC + '_%d'%i].get_out_ops_in_mode({
        self.submods[ENC + '_%d'%i].InKey.FT: inputs,
        self.submods[ENC + '_%d'%i].InKey.MASK: in_ops[self.InKey.MASK],
        self.submods[ENC + '_%d'%i].InKey.IS_TRN: in_ops[self.InKey.IS_TRN],
      }, mode)
      inputs = out_ops[self.submods[ENC + '_%d'%i].OutKey.OUTPUT]

    outputs = out_ops[self.submods[ENC + '_%d'%i].OutKey.OUTPUT] # (None, num_step, dim_hidden)

    with tf.variable_scope(self.name_scope):
      for fc_W, fc_B in zip(self.fc_Ws, self.fc_Bs):
        outputs = tf.nn.conv1d(outputs, fc_W, 1, 'VALID')
        outputs = tf.nn.bias_add(outputs, fc_B)
        outputs = tf.nn.relu(outputs)
      logits = tf.nn.conv1d(outputs, self.sigmoid_W, 1, 'VALID')
      logits = tf.nn.bias_add(logits, self.sigmoid_B)
      predicts = tf.nn.sigmoid(logits) # (None, num_step, num_pos_class)

      return {
        self.OutKey.LOGIT: logits,
        self.OutKey.PREDICT: predicts,
      }

  def _add_loss(self):
    with tf.variable_scope(self.name_scope):
      labels = self._inputs[self.InKey.LABEL]
      label_masks = self._inputs[self.InKey.LABEL_MASK]
      logits = self._outputs[self.OutKey.LOGIT]

      labels = tf.reshape(labels, (-1, self._config.num_pos_class))
      label_masks = tf.reshape(label_masks, (-1, self._config.num_pos_class))
      logits = tf.reshape(logits, (-1, self._config.num_pos_class))
      if self._config.focal_loss:
        loss = op_util.sigmoid_focal_loss(labels, logits)
      else:
        loss = tf.nn.sigmoid_cross_entropy_with_logits(
          labels=labels, logits=logits)
      loss = tf.reduce_sum(loss * label_masks) / tf.reduce_sum(label_masks)

    return loss

  def op_in_val(self, **kwargs):
    return {
      self.OutKey.PREDICT: self._outputs[self.OutKey.PREDICT],
    }

  def op_in_tst(self, **kwargs):
    return {
      self.OutKey.PREDICT: self._outputs[self.OutKey.PREDICT],
    }


TrnTst = birnn.TrnTst
PathCfg = birnn.PathCfg
Reader = birnn.Reader
ValReader = birnn.ValReader
