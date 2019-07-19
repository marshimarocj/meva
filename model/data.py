import tensorflow as tf
import numpy as np


def parse_record(record, mean):
  example = tf.train.Example()
  example.ParseFromString(record)

  ft = example.features.feature['fts'].bytes_list.value[0]
  shape = example.features.feature['fts.shape'].int64_list.value
  shape = [d for d in shape]
  ft = np.fromstring(ft, dtype=np.float32)
  ft = ft.reshape(shape)
  if mean and len(ft.shape) > 2:
    ft = np.mean(ft, (1, 2))

  label = example.features.feature['labels'].bytes_list.value[0]
  shape = example.features.feature['labels.shape'].int64_list.value
  shape = [d for d in shape]
  label = np.fromstring(label, dtype=np.float32)
  label = label.reshape(shape)

  label_mask = np.ones(label.shape[:1], dtype=np.float32)

  return ft, label, label_mask


def pad_data(fts, labels, label_masks, start, num_step):
  ft = fts[start:start + num_step]
  mask = np.ones((ft.shape[0],), dtype=np.float32)
  label = labels[start:start + num_step]
  label_mask = label_masks[start:start + num_step]
  end = ft.shape[0]
  if ft.shape[0] < num_step:
    num_fill = num_step - ft.shape[0]
    ft_fill = np.zeros((num_fill,) + ft.shape[1:], dtype=np.float32)
    ft = np.concatenate([ft, ft_fill], axis=0)

    mask_fill = np.zeros((num_fill,), dtype=np.float32)
    mask = np.concatenate([mask, mask_fill])

    label_fill = np.zeros((num_fill,) + label.shape[1:], dtype=np.float32)
    label = np.concatenate([label, label_fill], axis=0)

    label_mask_fill = np.zeros((num_fill,) + label_mask.shape[1:], dtype=np.float32)
    label_mask = np.concatenate([label_mask, label_mask_fill], axis=0)

  return ft, mask, label, label_mask
