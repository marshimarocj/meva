import os
import json
import subprocess

import numpy as np
from sklearn.metrics import roc_curve


'''func
'''
def select_best_epoch(log_dir, lower=-1, upper=-1, metric='mAP'):
  names = os.listdir(log_dir)
  max_mAP = 0.
  best_epoch = -1
  for name in names:
    if 'val_metrics' not in name:
      continue
    data = name.split('.')
    epoch = int(data[1])
    if (lower >= 0 and epoch < lower) or (upper >= 0 and epoch < upper):
      continue

    file = os.path.join(log_dir, name)
    with open(file) as f:
      data = json.load(f)
      mAP = data[metric]
      if mAP > max_mAP:
        max_mAP = mAP
        best_epoch = epoch

  return best_epoch, max_mAP


def gen_script_and_run(python_file, model_cfg_file, path_cfg_file, best_epoch, gpuid, **kwargs):
  cmd = [
    'python', os.path.join('../driver', python_file),
    model_cfg_file, path_cfg_file, 
    '--is_train', '0',
    '--best_epoch', str(best_epoch),
  ]
  for key in kwargs:
    cmd += ['--' + key, str(kwargs[key])]
  env = os.environ
  env['CUDA_VISIBLE_DEVICES'] = str(gpuid)
  p = subprocess.Popen(cmd, env=env)
  return p


'''expr
'''
def predict():
  root_dir = '/home/chenj/data' # light-1
  # root_dir = '/home/jiac/ssd/meva' # gpu9

  expr_name = os.path.join(root_dir, 'expr/of.128.1.128')
  python_file = 'birnn.py'

  gpuid = 1
  tst_dirs = [
    os.path.join(root_dir, 'compile', 'val'),
    os.path.join(root_dir, 'compile', 'val_neg'),
  ]
  out_name = 'val'

  model_cfg_file = '%s.model.json'%expr_name
  path_cfg_file = '%s.path.json'%expr_name
  log_dir = os.path.join(expr_name, 'log')

  best_epoch = select_best_epoch(log_dir)
  print best_epoch
  p = gen_script_and_run(python_file, model_cfg_file, path_cfg_file, best_epoch, gpuid,
    out_name=out_name, data_dirs=','.join(tst_dirs))
  p.wait()


def threshold():
  root_dir = '/home/chenj/data'

  rfa = 1.
  FT_STRIDE = 8
  FPS = 30
  target_fpr = rfa / 60 / FPS * FT_STRIDE

  expr_name = os.path.join(root_dir, 'expr/of.128.1.128')
  out_file = os.path.join(expr_name, 'pred', 'threshold.%.2f.json'%rfa)

  pred_file = os.path.join(expr_name, 'pred', 'val.npz')
  data = np.load(pred_file)
  predicts = data['predicts']
  labels = data['labels']

  num_label = predicts.shape[-1]
  out = []
  for l in range(num_label):
    predict = predicts[:, l]
    label = labels[:, l]
    fprs, tprs, thresholds = roc_curve(label, predict)
    i = 0
    for fpr, tpr, threshold in zip(fprs, tprs, thresholds):
      i += 1
      if fpr >= target_fpr:
        break
    tpr = np.mean(tprs[:i])
    out.append({
      'fpr': float(fpr),
      'tpr': float(tpr),
      'threshold': float(threshold),
    })

  with open(out_file, 'w') as fout:
    json.dump(out, fout, indent=2)


def report():
  root_dir = '/data/jiac/meva' # diva

  expr_name = os.path.join(root_dir, 'expr/conv_transformer/of.256.8.1.128.49')

  model_cfg_file = '%s.model.json'%expr_name
  path_cfg_file = '%s.path.json'%expr_name
  log_dir = os.path.join(expr_name, 'log')

  best_epoch, mAP = select_best_epoch(log_dir)
  print best_epoch, mAP


if __name__ == '__main__':
  # predict()
  # threshold()
  report()
