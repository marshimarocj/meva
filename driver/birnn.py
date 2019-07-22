import argparse
import sys
import os
import datetime
import cPickle
sys.path.append('../')

import numpy as np

import model.birnn
import common

ENC = model.birnn.RNN

def build_parser():
  parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

  parser.add_argument('model_cfg_file')
  parser.add_argument('path_cfg_file')
  parser.add_argument('--is_train', dest='is_train', type=int, default=True)
  parser.add_argument('--memory_fraction', dest='memory_fraction', type=float, default=1.0)
  # only in tst
  parser.add_argument('--best_epoch', dest='best_epoch', type=int, default=True)
  parser.add_argument('--data_dirs', default='')
  parser.add_argument('--out_name', default='')

  return parser


def load_and_fill_model_cfg(model_cfg_file, path_cfg):
  model_cfg = model.birnn.ModelConfig()
  model_cfg.load(model_cfg_file)

  return model_cfg


if __name__ == '__main__':
  parser = build_parser()
  opts = parser.parse_args()

  path_cfg = model.birnn.PathCfg()
  common.gen_dir_struct_info(path_cfg, opts.path_cfg_file)
  model_cfg = load_and_fill_model_cfg(opts.model_cfg_file, path_cfg)

  m = model.birnn.Model(model_cfg)

  if opts.is_train:
    with open(os.path.join(path_cfg.log_dir, 'cfg.pkl'), 'w') as fout:
      cPickle.dump(model_cfg, fout)
      cPickle.dump(path_cfg, fout)
      cPickle.dump(opts, fout)

    trntst = model.birnn.TrnTst(model_cfg, path_cfg, m)

    trn_reader = model.birnn.Reader(
      path_cfg.label2lid_file, path_cfg.trn_dirs, model_cfg.subcfgs[ENC].num_step, model_cfg.shift, shuffle=True)
    val_reader = model.birnn.ValReader(
      path_cfg.label2lid_file, [path_cfg.val_dir], model_cfg.subcfgs[ENC].num_step)
    if path_cfg.model_file != '':
      trntst.train(trn_reader, val_reader, memory_fraction=opts.memory_fraction, resume=True)
    else:
      trntst.train(trn_reader, val_reader, memory_fraction=opts.memory_fraction)
