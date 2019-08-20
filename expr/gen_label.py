import os
import json

from diva_common.structure.annotation import *


'''func
'''
def load_actvid_from_json(file):
  with open(file) as f:
    data = json.load(f)
  actv = ActvIdType()
  for eid in data:
    meta = EventMeta()
    meta.eid = int(eid)
    meta.parse(data[eid])
    actv.eid2event_meta[eid] = meta
  return actv


'''expr
'''
def tst_load_actvid_from_json():
  root_dir = '/mnt/sda/jiac'
  file = os.path.join(root_dir, 'f330_train_annotation', 'teamB', '2018-03-09_10-15-01_10-20-01_bus_G505.json')

  actv = load_actvid_from_json(file)


if __name__ == '__main__':
  tst_load_actvid_from_json()
