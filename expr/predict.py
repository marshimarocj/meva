import os
import json


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

  return best_epoch


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
  root_dir = '/home/chenj/data'

  expr_name = os.path.join(root_dir, 'expr/of.128.1.128')
  python_file = 'birnn.py'

  gpuid = 0
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


if __name__ == '__main__':
  predict()
