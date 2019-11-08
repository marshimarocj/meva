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
import op_util
import birnn


class ModelConfig(framework.model.module.ModelConfig):
  def __init__(self):
    framework.model.module.ModelConfig.__init__(self)

    self.num_pos_class = 12
    self.dim_ft = 1024
    self.dim_hiddens = []
    self.focal_loss = False
    self.num_step = 64

  def _assert(self):
    pass


def gen_cfg(**kwargs):
  cfg = ModelConfig()
  cfg.val_iter = -1
  cfg.monitor_iter = 50
  cfg.trn_batch_size = 32
  cfg.tst_batch_size = 32
  cfg.base_lr = 1e-4
  cfg.num_epoch = 100
  cfg.dim_ft = kwargs['dim_ft']
  cfg.num_pos_class = kwargs['num_pos_class']
  cfg.dim_hiddens = kwargs['dim_hiddens']
  cfg.focal_loss = kwargs['focal_loss']
  cfg.val_loss = False

  return cfg


class Model(framework.model.module.AbstractModel):
  name_scope = 'vanilla.Model'

  class InKey(enum.Enum):
    FT = 'fts'
    IS_TRN = 'is_training'
    LABEL = 'label'
    LABEL_MASK = 'label_mask'

  class OutKey(enum.Enum):
    LOGIT = 'logit'
    PREDICT = 'predict'

  def _set_submods(self):
    return {}

  def _add_input_in_mode(self, mode):
    with tf.variable_scope(self.name_scope):
      fts = tf.placeholder(
        tf.float32, shape=(None, self._config.num_step, self._config.dim_ft), name=self.InKey.FT.value)
      is_training = tf.placeholder(
        tf.bool, shape=(),name=self.InKey.IS_TRN.value)
      if mode == framework.model.module.Mode.TRN_VAL:
        label = tf.placeholder(
          tf.float32, shape=(None, self._config.num_step, self._config.num_pos_class), name=self.InKey.LABEL.value)
        label_mask = tf.placeholder(
          tf.float32, shape=(None, self._config.num_step, self._config.num_pos_class), name=self.InKey.LABEL_MASK.value)
        return {
          self.InKey.FT: fts,
          self.InKey.LABEL: label,
          self.InKey.LABEL_MASK: label_mask,
          self.InKey.IS_TRN: is_training,
        }
      else:
        return {
          self.InKey.FT: fts,
          self.InKey.IS_TRN: is_training,
        }

  def _build_parameter_graph(self):
    self.fc_Ws = []
    self.fc_Bs = []
    with tf.variable_scope(self.name_scope):
      dim_inputs = [self._config.dim_ft] + self._config.dim_hiddens[:-1]
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

      if len(dim_outputs) > 0:
        dim_output = dim_outputs[-1]
      else:
        dim_output = dim_inputs[0]
      self.sigmoid_W = tf.contrib.framework.model_variable('sigmoid_W',
        shape=(1, dim_output, self._config.num_pos_class), dtype=tf.float32,
        initializer=tf.contrib.layers.xavier_initializer())
      self.sigmoid_B = tf.contrib.framework.model_variable('sigmoid_B',
        shape=(self._config.num_pos_class,), dtype=tf.float32,
        initializer=tf.constant_initializer(0.))
      self._weights.append(self.sigmoid_W)
      self._weights.append(self.sigmoid_B)

  def get_out_ops_in_mode(self, in_ops, mode):
    fts = in_ops[self.InKey.FT]
    with tf.variable_scope(self.name_scope):
      tst_outputs = fts
      for fc_W, fc_B in zip(self.fc_Ws, self.fc_Bs):
        tst_outputs = tf.nn.conv1d(tst_outputs, fc_W, 1, 'VALID')
        tst_outputs = tf.nn.bias_add(tst_outputs, fc_B)
        tst_outputs = tf.nn.relu(tst_outputs)
      tst_predicts = tf.nn.conv1d(tst_outputs, self.sigmoid_W, 1, 'VALID')
      tst_predicts = tf.nn.bias_add(tst_predicts, self.sigmoid_B)
      tst_predicts = tf.nn.sigmoid(tst_predicts) # (None, num_step, num_pos_class)

      if mode == framework.model.module.Mode.TRN_VAL:
        outputs = fts
        for fc_W, fc_B in zip(self.fc_Ws, self.fc_Bs):
          outputs = tf.nn.conv1d(outputs, fc_W, 1, 'VALID')
          outputs = tf.nn.bias_add(outputs, fc_B)
          outputs = tf.nn.relu(outputs)
        logits = tf.nn.conv1d(outputs, self.sigmoid_W, 1, 'VALID')
        logits = tf.nn.bias_add(logits, self.sigmoid_B)

        return {
          self.OutKey.LOGIT: logits,
          self.OutKey.PREDICT: tst_predicts,
        }
      else:
        return {
          self.OutKey.PREDICT: tst_predicts,
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


class TrnTst(birnn.TrnTst):
  def _construct_feed_dict_in_trn(self, data):
    return {
      self.model.inputs[self.model.InKey.FT]: data['fts'],
      self.model.inputs[self.model.InKey.IS_TRN]: True,
      self.model.inputs[self.model.InKey.LABEL]: data['labels'],
      self.model.inputs[self.model.InKey.LABEL_MASK]: data['label_masks'],
    }

  def _construct_feed_dict_in_val(self, data):
    return {
      self.model.inputs[self.model.InKey.FT]: data['fts'],
      self.model.inputs[self.model.InKey.IS_TRN]: False,
      self.model.inputs[self.model.InKey.LABEL]: data['labels'],
      self.model.inputs[self.model.InKey.LABEL_MASK]: data['label_masks'],
    }

  def _construct_feed_dict_in_tst(self, data):
    return {
      self.model.inputs[self.model.InKey.FT]: data['fts'],
      self.model.inputs[self.model.InKey.IS_TRN]: False,
    }


PathCfg = birnn.PathCfg
Reader = birnn.Reader
ValReader = birnn.ValReader
