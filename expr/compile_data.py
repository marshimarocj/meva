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
    os.path.join(root_dir, 'meva_train', 'feature_trn_teamA.json'),
    os.path.join(root_dir, 'meva_train', 'feature_trn_teamB.json'),
    os.path.join(root_dir, 'meva_train', 'feature_trn_teamC.json'),
  ]
  out_files = [
    os.path.join(root_dir, 'compile', 'all', 'teamA.tfrecord'),
    os.path.join(root_dir, 'compile', 'all', 'teamB.tfrecord'),
    os.path.join(root_dir, 'compile', 'all', 'teamC.tfrecord'),
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
        'fts': framework.util.io.bytes_feature([fts.tostring()]),
        'fts.shape': framework.util.io.int64_feature(fts.shape),
        'labels': framework.util.io.bytes_feature([labels.tostring()]),
        'labels.shape': framework.util.io.int64_feature(labels.shape),
      }))
      records.append(example)

    options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)

    with tf.python_io.TFRecordWriter(out_file, options=options) as writer:
      meta_record = framework.util.io.meta_record(len(records))
      writer.write(meta_record.SerializeToString())
      for record in records:
        writer.write(record.SerializeToString())


if __name__ == '__main__':
  compile_data2tfrecord()
