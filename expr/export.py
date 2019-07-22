import os
import sys
import json
sys.path.append('../')

import tensorflow as tf

import model.birnn


'''func
'''


'''expr
'''
def export_birnn_model():
  root_dir = '/home/chenj/data' # light-1
  expr_dir = os.path.join(root_dir, 'expr', 'of.128.1.128')
  model_cfg_file = '%s.model.json'%expr_dir
  path_cfg_file = '%s.path.json'%expr_dir
  model_file = os.path.join(expr_dir, 'model', 'epoch-98')
  out_dir = os.path.join(expr_dir, 'deliver')

  model_cfg = model.birnn.ModelConfig()
  model_cfg.load(model_cfg_file)

  path_cfg = model.birnn.PathCfg()
  driver.common.gen_dir_struct_info(path_cfg, path_cfg_file)
  path_cfg.model_file = model_file

  m = model.birnn.Model(model_cfg)

  tst_graph = m.build_tst_graph()
  print m._outputs[m.OutKey.PREDICT].name
  gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.)
  config_proto = tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)
  with tf.Session(graph=tst_graph, config=config_proto) as sess:
    sess.run(m.init_op)
    m.saver.restore(sess, path_cfg.model_file)

    inputs = {
      'fts': m._inputs[m.InKey.FT],
      'masks': m._inputs[m.InKey.MASK],
      'is_trn': m._inputs[m.InKey.IS_TRN],
    }
    outputs = {
      'predicts': m._outputs[m.OutKey.PREDICT],
    }
    tf.saved_model.simple_save(sess, out_dir, inputs, outputs)


if __name__ == '__main__':
  export_birnn_model()
