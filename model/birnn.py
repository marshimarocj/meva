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
import framework.impl.encoder.birnn
import op_util
import data

ENC = 'pca'
RNN = 'rnn'
CELL = framework.impl.encoder.birnn.CELL
RCELL = framework.impl.encoder.birnn.RCELL


class ModelConfig(framework.model.module.ModelConfig):
  def __init__(self):
    framework.model.module.ModelConfig.__init__(self)

    self.subcfgs[RNN] = framework.impl.encoder.birnn.Config()
    self.subcfgs[ENC] = framework.impl.encoder.pca.Config()
    self.num_pos_class = 12
    self.dim_hiddens = []
    self.focal_loss = False
    self.shift = 4

  def _assert(self):
    rnn_cfg = self.subcfgs[RNN]
    assert rnn_cfg.subcfgs[CELL].dim_input == self.subcfgs[ENC].dim_output
    assert rnn_cfg.subcfgs[RCELL].dim_input == self.subcfgs[ENC].dim_output


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
  cfg.focal_loss = kwargs['focal_loss']
  cfg.val_loss = False
  cfg.shift = kwargs['shift']

  rnn_cfg = cfg.subcfgs[RNN]
  rnn_cfg.num_step = kwargs['num_step']
  rnn_cfg.cell_type = kwargs['cell']

  cell_cfg = rnn_cfg.subcfgs[CELL]
  cell_cfg.dim_hidden = kwargs['cell_dim_hidden']
  cell_cfg.dim_input = kwargs['dim_embed']
  cell_cfg.keepout_prob = 0.5
  cell_cfg.keepin_prob = 0.5

  cell_cfg = rnn_cfg.subcfgs[RCELL]
  cell_cfg.dim_hidden = kwargs['cell_dim_hidden']
  cell_cfg.dim_input = kwargs['dim_embed']
  cell_cfg.keepout_prob = 0.5
  cell_cfg.keepin_prob = 0.5

  encoder_cfg = cfg.subcfgs[ENC]
  encoder_cfg.dim_ft = kwargs['dim_ft']
  encoder_cfg.dim_output = kwargs['dim_embed']

  return cfg


class Model(framework.model.module.AbstractModel):
  name_scope = 'rnn.Model'

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
    return {
      RNN: framework.impl.encoder.birnn.Encoder(self._config.subcfgs[RNN]),
      ENC: framework.impl.encoder.pca.Encoder1D(self._config.subcfgs[ENC]),
    }

  def _add_input_in_mode(self, mode):
    rnn_cfg = self._config.subcfgs[RNN]
    pca_cfg = self._config.subcfgs[ENC]
    with tf.variable_scope(self.name_scope):
      fts = tf.placeholder(
        tf.float32, shape=(None, rnn_cfg.num_step, pca_cfg.dim_ft), name=self.InKey.FT.value)
      masks = tf.placeholder(
        tf.float32, shape=(None, rnn_cfg.num_step), name=self.InKey.MASK.value)
      is_training = tf.placeholder(
        tf.bool, shape=(),name=self.InKey.IS_TRN.value)
      if mode == framework.model.module.Mode.TRN_VAL:
        label = tf.placeholder(
          tf.float32, shape=(None, rnn_cfg.num_step, self._config.num_pos_class), name=self.InKey.LABEL.value)
        label_mask = tf.placeholder(
          tf.float32, shape=(None, rnn_cfg.num_step, self._config.num_pos_class), name=self.InKey.LABEL_MASK.value)
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
    rnn_cfg = self._config.subcfgs[RNN]
    self.fc_Ws = []
    self.fc_Bs = []
    with tf.variable_scope(self.name_scope):
      dim_input = rnn_cfg.subcfgs[CELL].dim_hidden + rnn_cfg.subcfgs[RCELL].dim_hidden
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
    out_ops = self.submods[ENC].get_out_ops_in_mode({
      self.submods[ENC].InKey.FT: in_ops[self.InKey.FT],
    }, mode)

    with tf.variable_scope(self.name_scope):
      shape = tf.shape(in_ops[self.InKey.FT])
      init_state = tf.zeros((shape[0], self._config.subcfgs[RNN].subcfgs[CELL].dim_hidden))

    out_ops = self.submods[RNN].get_out_ops_in_mode({
      self.submods[RNN].InKey.FT: out_ops[self.submods[ENC].OutKey.EMBED],
      self.submods[RNN].InKey.MASK: in_ops[self.InKey.MASK],
      self.submods[RNN].InKey.IS_TRN: in_ops[self.InKey.IS_TRN],
      self.submods[RNN].InKey.INIT_STATE: init_state,
    }, mode)

    tst_outputs = out_ops[self.submods[RNN].OutKey.TST_OUTPUT] # (None, num_step, dim_hidden)
    if mode == framework.model.module.Mode.TRN_VAL:
      outputs = out_ops[self.submods[RNN].OutKey.OUTPUT]

    with tf.variable_scope(self.name_scope):
      for fc_W, fc_B in zip(self.fc_Ws, self.fc_Bs):
        tst_outputs = tf.nn.conv1d(tst_outputs, fc_W, 1, 'VALID')
        tst_outputs = tf.nn.bias_add(tst_outputs, fc_B)
        tst_outputs = tf.nn.relu(tst_outputs)
      tst_predicts = tf.nn.conv1d(tst_outputs, self.sigmoid_W, 1, 'VALID')
      tst_predicts = tf.nn.bias_add(tst_predicts, self.sigmoid_B)
      tst_predicts = tf.nn.sigmoid(tst_predicts) # (None, num_step, num_pos_class)

      if mode == framework.model.module.Mode.TRN_VAL:
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


class TrnTst(framework.model.trntst.TrnTst):
  def feed_data_and_run_loss_op_in_val(self, data, sess):
    op_dict = self.model.op_in_val()

    feed_dict = self._construct_feed_dict_in_val(data)
    loss = sess.run(op_dict[self.model.DefaultKey.LOSS], feed_dict=feed_dict)

    return loss

  def _construct_feed_dict_in_trn(self, data):
    return {
      self.model.inputs[self.model.InKey.FT]: data['fts'],
      self.model.inputs[self.model.InKey.MASK]: data['masks'],
      self.model.inputs[self.model.InKey.IS_TRN]: True,
      self.model.inputs[self.model.InKey.LABEL]: data['labels'],
      self.model.inputs[self.model.InKey.LABEL_MASK]: data['label_masks'],
    }

  def _construct_feed_dict_in_val(self, data):
    return {
      self.model.inputs[self.model.InKey.FT]: data['fts'],
      self.model.inputs[self.model.InKey.MASK]: data['masks'],
      self.model.inputs[self.model.InKey.IS_TRN]: False,
      self.model.inputs[self.model.InKey.LABEL]: data['labels'],
      self.model.inputs[self.model.InKey.LABEL_MASK]: data['label_masks'],
    }

  def _construct_feed_dict_in_tst(self, data):
    return {
      self.model.inputs[self.model.InKey.FT]: data['fts'],
      self.model.inputs[self.model.InKey.MASK]: data['masks'],
      self.model.inputs[self.model.InKey.IS_TRN]: False,
    }

  def predict_and_eval_in_val(self, sess, tst_reader, metrics):
    label2lid = tst_reader.label2lid
    lid2label = {}
    for label in label2lid:
      lid2label[label2lid[label]] = label
    num_label = len(label2lid)

    op_dict = self.model.op_in_val()
    tst_batch_size = self.model_cfg.tst_batch_size

    all_predicts = []
    all_labels = []
    for data in tst_reader.yield_tst_batch(tst_batch_size):
      label_masks = data['label_masks']
      labels = data['labels']
      feed_dict = self._construct_feed_dict_in_val(data)
      predicts = sess.run(op_dict[self.model.OutKey.PREDICT], feed_dict=feed_dict)
      for predict, label_mask, label in zip(predicts, label_masks, labels):
        num = label_mask.shape[0]
        for i in range(num):
          if np.sum(label_mask[i]) != num_label:
            continue
          all_predicts.append(predict[i])
          all_labels.append(label[i])
    all_predicts = np.array(all_predicts)
    all_labels = np.array(all_labels)

    mAP = 0.
    for l in range(num_label):
      score = average_precision_score(all_labels[:, l], all_predicts[:, l])
      metrics[lid2label[l]] = score
      mAP += score
    metrics['mAP'] = mAP / num_label

    all_predicts = all_predicts.reshape((-1,))
    all_labels = all_labels.reshape((-1,))
    score = average_precision_score(all_labels, all_predicts)
    metrics['GAP'] = score

  def predict_in_tst(self, sess, tst_reader, predict_file):
    op_dict = self.model.op_in_val()
    tst_batch_size = self.model_cfg.tst_batch_size

    all_predicts = []
    all_labels = []
    for data in tst_reader.yield_tst_batch(tst_batch_size):
      label_masks = data['label_masks']
      labels = data['labels']
      feed_dict = self._construct_feed_dict_in_tst(data)
      predicts = sess.run(op_dict[self.model.OutKey.PREDICT], feed_dict=feed_dict)
      j = 0
      for predict, label_mask, label in zip(predicts, label_masks, labels):
        num = label_mask.shape[0]
        for i in range(num):
          if np.sum(label_mask[i]) != label[i].shape[-1]:
            continue
          all_predicts.append(predict[i])
          all_labels.append(label[i])
        j += 1
    all_predicts = np.array(all_predicts)
    all_labels = np.array(all_labels)

    np.savez_compressed(predict_file, predicts=all_predicts, labels=all_labels)


class PathCfg(framework.model.trntst.PathCfg):
  def __init__(self):
    framework.model.trntst.PathCfg.__init__(self)

    self.trn_dirs = []
    self.val_dir = ''
    self.label2lid_file = ''

    self.output_dir = ''
    self.log_file = ''


class Reader(framework.model.data.Reader):
  def __init__(self, label2lid_file, data_dirs, num_step, shift, shuffle=False):
    self.label2lid = {}
    self.shuffle = shuffle
    self.num_step = num_step
    self.shift = shift

    self.fts = []
    self.labels = []
    self.masks = []

    with open(label2lid_file) as f:
      self.label2lid = json.load(f)

    for data_dir in data_dirs:
      names = os.listdir(data_dir)
      names = sorted(names)

      options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)
      for name in names:
        print name
        chunk_file = os.path.join(data_dir, name)
        record_iterator = tf.python_io.tf_record_iterator(path=chunk_file, options=options)
        record_iterator.next()
        for record in record_iterator:
          ft, label, label_mask = data.parse_record(record, True)
          self.fts.append(ft)
          self.labels.append(label)
          self.masks.append(label_mask)

    self.total = len(self.fts)
    self.idxs = range(self.total)

  def reset(self):
    if self.shuffle:
      random.shuffle(self.idxs)

  def num_record(self):
    return self.total # pseudo

  def yield_trn_batch(self, batch_size):
    batch_fts = []
    batch_masks = []
    batch_labels = []
    batch_label_masks = []
    for idx in self.idxs:
      ft = self.fts[idx]
      label = self.labels[idx]
      label_mask = self.masks[idx]
      num = ft.shape[0]
      i = 0
      while i < num:
        if i > 0 and num-i < self.num_step/2:
          break
        pad_ft, pad_mask, pad_label, pad_label_mask = data.pad_data(ft, label, label_mask, i, self.num_step)
        if np.sum(pad_label_mask) == 0. or np.sum(pad_mask) == 0.:
          continue
        batch_fts.append(pad_ft)
        batch_masks.append(pad_mask)
        batch_labels.append(pad_label)
        batch_label_masks.append(pad_label_mask)
        i += self.shift

        if len(batch_fts) == batch_size:
          yield {
            'fts': batch_fts,
            'masks': batch_masks,
            'labels': batch_labels,
            'label_masks': batch_label_masks,
          }
          batch_fts = []
          batch_masks = []
          batch_labels = []
          batch_label_masks = []
    if len(batch_fts) > 0:
      yield {
        'fts': batch_fts,
        'masks': batch_masks,
        'labels': batch_labels,
        'label_masks': batch_label_masks,
      }

  def yield_tst_batch(self, batch_size):
    for data in self.yield_trn_batch(batch_size):
      yield data


class ValReader(framework.model.data.Reader):
  def __init__(self, label2lid_file, data_dirs, num_step):
    self.label2lid = {}
    self.shuffle = False
    self.num_step = num_step

    self.fts = []
    self.labels = []
    self.masks = []

    with open(label2lid_file) as f:
      self.label2lid = json.load(f)

    for data_dir in data_dirs:
      names = os.listdir(data_dir)
      names = sorted(names)

      options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)
      for name in names:
        print name
        chunk_file = os.path.join(data_dir, name)
        record_iterator = tf.python_io.tf_record_iterator(path=chunk_file, options=options)
        record_iterator.next()
        for record in record_iterator:
          ft, label, label_mask = data.parse_record(record, True)
          self.fts.append(ft)
          self.labels.append(label)
          self.masks.append(label_mask)

    self.total = len(self.fts)
    self.idxs = range(self.total)

  def reset(self):
    if self.shuffle:
      random.shuffle(self.idxs)

  def num_record(self):
    return self.total # pseudo

  def yield_tst_batch(self, batch_size):
    batch_fts = []
    batch_masks = []
    batch_labels = []
    batch_label_masks = []
    for idx in self.idxs:
      ft = self.fts[idx]
      label = self.labels[idx]
      label_mask = self.masks[idx]
      num = ft.shape[0]
      if self.shuffle:
        i = random.randint(0, min(max(num-1, 0), self.num_step/2))
      else:
        i = 0
      while i < num:
        pad_ft, pad_mask, pad_label, pad_label_mask = data.pad_data(ft, label, label_mask, i, self.num_step)
        batch_fts.append(pad_ft)
        batch_masks.append(pad_mask)
        batch_labels.append(pad_label)
        batch_label_masks.append(pad_label_mask)
        i += self.num_step

        if len(batch_fts) == batch_size:
          yield {
            'fts': batch_fts,
            'masks': batch_masks,
            'labels': batch_labels,
            'label_masks': batch_label_masks,
          }
          batch_fts = []
          batch_masks = []
          batch_labels = []
          batch_label_masks = []

        if i > 0 and num-i < self.num_step/2:
          break
    if len(batch_fts) > 0:
      yield {
        'fts': batch_fts,
        'masks': batch_masks,
        'labels': batch_labels,
        'label_masks': batch_label_masks,
      }
