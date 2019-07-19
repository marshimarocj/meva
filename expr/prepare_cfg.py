import os
import sys
import json
sys.path.append('../')

import model.birnn


'''func
'''


'''expr
'''
def prepare_birnn():
  root_dir = '/home/chenj/data' # light-1
  trn_data_dirs = [
    os.path.join(root_dir, 'compile', 'trn')
  ]
  val_data_dir = os.path.join(root_dir, 'compile', 'val')
  expr_dir = os.path.join(root_dir, 'expr')

  params = {
    'dim_hiddens': [128],
    'dim_embed': 128,
    'num_step': 128,
    'num_pos_class': 19,
    'dim_ft': 1024,
    'shift': 64/2,
    'focal_loss': True,
    'cell': 'lstm',
    'cell_dim_hidden': 128,
  }

  outprefix = '%s/%s.%s.%d.%s.%d'%(
    expr_dir, ft_name, '_'.join([str(d) for d in params['dim_hiddens']]), 
    params['focal_loss'], '-'.join(props_types), 
    params['num_step']
  )
  model_cfg_file = '%s.model.json'%outprefix
  cfg = model.birnn.gen_cfg(**params)
  cfg.num_epoch = 100
  cfg.save(model_cfg_file)

  path_cfg = {
    'trn_dirs': trn_data_dirs,
    'val_dir': val_data_dir,
    'output_dir': outprefix,
    'label2lid_file': os.path.join(root_dir, 'lst', 'label2lid_%d.json'%params['num_pos_class'])
  }
  path_cfg_file = '%s.path.json'%outprefix
  with open(path_cfg_file, 'w') as fout:
    json.dump(path_cfg, fout, indent=2)

  if not os.path.exists(outprefix):
    os.mkdir(outprefix)


if __name__ == '__main__':
  prepare_birnn()