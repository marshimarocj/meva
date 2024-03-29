import os
import sys
import json
sys.path.append('../')

import model.birnn
import model.vanilla
import model.conv_transformer
import model.multi_conv_transformer


'''func
'''


'''expr
'''
def prepare_birnn():
  # root_dir = '/home/chenj/data' # light-1
  # root_dir = '/home/jiac/ssd/meva' # gpu9
  root_dir = '/data/jiac/meva'
  trn_data_dirs = [
    os.path.join(root_dir, 'compile', 'trn')
  ]
  val_data_dir = os.path.join(root_dir, 'compile', 'val')
  expr_dir = os.path.join(root_dir, 'expr', 'birnn')
  ft_name = 'of'

  params = {
    'dim_hiddens': [128],
    'dim_embed': 128,
    'num_step': 128,
    'num_pos_class': 35,
    'dim_ft': 1024,
    'shift': 64/2,
    'focal_loss': True,
    'cell': 'lstm',
    'cell_dim_hidden': 128,
  }

  outprefix = '%s/%s.%s.%d.%d'%(
    expr_dir, ft_name, '_'.join([str(d) for d in params['dim_hiddens']]), 
    params['focal_loss'], params['num_step']
  )
  model_cfg_file = '%s.model.json'%outprefix
  cfg = model.birnn.gen_cfg(**params)
  cfg.num_epoch = 10000
  cfg.save(model_cfg_file)

  path_cfg = {
    'trn_dirs': trn_data_dirs,
    'val_dir': val_data_dir,
    'output_dir': outprefix,
    'label2lid_file': os.path.join(root_dir, 'label.json')
  }
  path_cfg_file = '%s.path.json'%outprefix
  with open(path_cfg_file, 'w') as fout:
    json.dump(path_cfg, fout, indent=2)

  if not os.path.exists(outprefix):
    os.mkdir(outprefix)


def prepare_vanilla():
  root_dir = '/home/chenj/data' # light-1
  trn_data_dirs = [
    os.path.join(root_dir, 'compile', 'trn')
  ]
  val_data_dir = os.path.join(root_dir, 'compile', 'val')
  expr_dir = os.path.join(root_dir, 'expr', 'reproduce')
  ft_name = 'of'

  params = {
    'dim_hiddens': [256],
    'num_step': 64,
    'num_pos_class': 35,
    'dim_ft': 1024,
    'focal_loss': False,
  }

  outprefix = '%s/%s.%s'%(
    expr_dir, ft_name, '_'.join([str(d) for d in params['dim_hiddens']]), 
  )
  model_cfg_file = '%s.model.json'%outprefix
  cfg = model.vanilla.gen_cfg(**params)
  cfg.num_epoch = 10000
  cfg.save(model_cfg_file)

  path_cfg = {
    'trn_dirs': trn_data_dirs,
    'val_dir': val_data_dir,
    'output_dir': outprefix,
    'label2lid_file': os.path.join(root_dir, 'meva_train', 'label.json')
  }
  path_cfg_file = '%s.path.json'%outprefix
  with open(path_cfg_file, 'w') as fout:
    json.dump(path_cfg, fout, indent=2)

  if not os.path.exists(outprefix):
    os.mkdir(outprefix)


def prepare_conv_transformer():
  root_dir = '/data/jiac/meva' # diva
  trn_data_dirs = [
    os.path.join(root_dir, 'compile_2', 'trn'),
  ]
  val_data_dir = os.path.join(root_dir, 'compile_2', 'val')
  expr_dir = os.path.join(root_dir, 'expr', 'conv_transformer')
  ft_name = 'of'

  params = {
    'num_pos_class': 35,
    'dim_hiddens': [256],
    'num_step': 64,
    # 'kernel_size': 33,
    'kernel_size': 9,
    'focal_loss': True,
    'num_head': 8,
    'dim_embed': 256,
    'dim_ft': 1024,
    'norms': False,
    'shift': 64/2,
  }

  outprefix = '%s/%s.%s.%d.%d.%d.%d'%(
    expr_dir, ft_name, '_'.join([str(d) for d in params['dim_hiddens']]), 
    params['num_head'], params['focal_loss'], 
    params['num_step'], params['kernel_size']
  )
  model_cfg_file = '%s.model.json'%outprefix
  cfg = model.conv_transformer.gen_cfg(**params)
  cfg.num_epoch = 100
  cfg.save(model_cfg_file)

  path_cfg = {
    'trn_dirs': trn_data_dirs,
    'val_dir': val_data_dir,
    'output_dir': outprefix,
    'label2lid_file': os.path.join(root_dir, 'label.json')
  }
  path_cfg_file = '%s.path.json'%outprefix
  with open(path_cfg_file, 'w') as fout:
    json.dump(path_cfg, fout, indent=2)

  if not os.path.exists(outprefix):
    os.mkdir(outprefix)


def prepare_multi_conv_transformer():
  root_dir = '/data/jiac/meva' # diva
  trn_data_dirs = [
    os.path.join(root_dir, 'compile_2', 'trn'),
  ]
  val_data_dir = os.path.join(root_dir, 'compile_2', 'val')
  expr_dir = os.path.join(root_dir, 'expr', 'multi_conv_transformer')
  ft_name = 'of'

  params = {
    'num_pos_class': 35,
    'dim_hiddens': [256],
    'num_step': 64,
    'kernel_size': 17,
    'focal_loss': True,
    'num_head': 8,
    'dim_embed': 256,
    'dim_ft': 1024,
    'norms': False,
    'shift': 64/2,
    'num_layer': 4,
  }

  outprefix = '%s/%s.%s.%d.%d.%d.%d.%d'%(
    expr_dir, ft_name, '_'.join([str(d) for d in params['dim_hiddens']]), 
    params['num_head'], params['focal_loss'], 
    params['num_step'], params['kernel_size'], params['num_layer']
  )
  model_cfg_file = '%s.model.json'%outprefix
  cfg = model.multi_conv_transformer.gen_cfg(**params)
  cfg.num_epoch = 100
  cfg.save(model_cfg_file)

  path_cfg = {
    'trn_dirs': trn_data_dirs,
    'val_dir': val_data_dir,
    'output_dir': outprefix,
    'label2lid_file': os.path.join(root_dir, 'label.json')
  }
  path_cfg_file = '%s.path.json'%outprefix
  with open(path_cfg_file, 'w') as fout:
    json.dump(path_cfg, fout, indent=2)

  if not os.path.exists(outprefix):
    os.mkdir(outprefix)


if __name__ == '__main__':
  # prepare_birnn()
  # prepare_vanilla()
  # prepare_conv_transformer()
  prepare_multi_conv_transformer()
