import os
import json

import tensorflow as tf
import numpy as np

import framework.util.io


'''func
'''


'''expr
'''
def compile_data2tfrecord():
  root_dir = '/home/chenj/data'
  label_file = os.path.join(root_dir, 'meva_train', 'label.json')
  data_files = [
    # os.path.join(root_dir, 'meva_train', 'feature_trn_teamA.json'),
    # os.path.join(root_dir, 'meva_train', 'feature_trn_teamB.json'),
    # os.path.join(root_dir, 'meva_train', 'feature_trn_teamC.json'),
    os.path.join(root_dir, 'meva_train', 'feature_f330_train.json'),
    os.path.join(root_dir, 'meva_train', 'feature_f330_valid.json'),
  ]
  out_files = [
    # os.path.join(root_dir, 'compile', 'all', 'teamA.tfrecord'),
    # os.path.join(root_dir, 'compile', 'all', 'teamB.tfrecord'),
    # os.path.join(root_dir, 'compile', 'all', 'teamC.tfrecord'),
    os.path.join(root_dir, 'compile', 'trn', '0.tfrecord'),
    os.path.join(root_dir, 'compile', 'val', '0.tfrecord'),
  ]

  with open(label_file) as f:
    label2lid = json.load(f)
  num_label = len(label2lid)

  for data_file, out_file in zip(data_files, out_files):
    with open(data_file) as f:
      data = json.load(f)

    records = []
    for name in data:
      ft = data[name]
      for label in label2lid:
        if label in name:
          lid = label2lid[label]
      ft = np.array(ft, dtype=np.float32)
      num_ft = ft.shape[0]
      label = np.zeros((num_ft, num_label), dtype=np.float32)
      label[:, lid] = 1.
      print lid

      example = tf.train.Example(features=tf.train.Features(feature={
        'fts': framework.util.io.bytes_feature([ft.tostring()]),
        'fts.shape': framework.util.io.int64_feature(ft.shape),
        'labels': framework.util.io.bytes_feature([label.tostring()]),
        'labels.shape': framework.util.io.int64_feature(label.shape),
      }))
      records.append(example)

    options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)

    with tf.python_io.TFRecordWriter(out_file, options=options) as writer:
      meta_record = framework.util.io.meta_record(len(records))
      writer.write(meta_record.SerializeToString())
      for record in records:
        writer.write(record.SerializeToString())


def gen_pos_lst():
  root_dir = '/home/chenj/data'
  lst_file = os.path.join(root_dir, 'meva_train', 'gt_proposals', 'valid.lst')
  out_file = os.path.join(root_dir, 'meva_train', 'gt_proposals', 'video2pos_eids.json')

  videos = []
  with open(lst_file) as f:
    for line in f:
      line = line.strip()
      name, _ = os.path.splitext(line)
      videos.append(name)

  video2pos_eids = {}
  for video in videos:
    video2pos_eids[video] = []

    teams = ['teamA', 'teamB', 'teamC']
    for team in teams:
      pos_lst_file = os.path.join(root_dir, 'meva_train', 'gt_proposals', team, video + '.json')
      with open(pos_lst_file) as f:
        data = json.load(f)
      for eid in data:
        video2pos_eids[video].append(eid)
    with open(out_file, 'w') as fout:
      json.dump(video2pos_eids, fout, indent=2)


def compile_neg_data():
  root_dir = '/home/chenj/data'
  lst_file = os.path.join(root_dir, 'meva_train', 'gt_proposals', 'valid.lst')



if __name__ == '__main__':
  # compile_data2tfrecord()
  gen_pos_lst()
