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


def gen_proposal_label():
  root_dir = '/mnt/sda/jiac'
  video = '2018-03-07_17-20-00_17-25-00_school_G330'
  gt_label_file = os.path.join(root_dir, 'f330_train_annotation', 'teamB', video + '.json')
  prop_file = os.path.join(root_dir, 'f330_train_fb_feat', 'trn', 'indoor', video + '.avi', 'annotation', video, 'actv_id_type.pkl')
  out_file = os.path.join('/home/chenj/data/label', video + '.pkl')

  STRIDE = 8 # frames

  gt_actv = load_actvid_from_json(gt_label_file)
  actv = ActvIdType()
  actv.load(prop_file)

  eid2label = {}
  for eid in actv.eid2event_meta:
    emeta = actv.eid2event_meta[eid]
    start_frame = emeta.start_frame
    end_frame = emeta.end_frame

    labels = []
    label_masks = []
    gt_eids = set()
    for f in range(start_frame + STRIDE/2, end_frame, STRIDE):
      label = np.zeros((num_label,))
      label_mask = np.ones((num_label,))
      for gt_eid in gt_actv.eid2event_meta:
        gt_emeta = gt_actv.eid2event_meta[gt_eid]
        if f >= gt_emeta.event_end or f < gt_emeta.event_begin:
          continue

        siou = calc_spatial_iou(emeta, gt_emeta, f-STRIDE/2, f+STRIDE/2)
        if siou >= SPATIAL_DELTA:
          lid = label2lid[gt_emeta.event]
          label[lid] = 1.
          gt_eids.add(gt_eid)
      labels.append(label)
      label_masks.append(label_mask)
    gt_eids = list(gt_eids)

    labels = np.array(labels, dtype=np.float32)
    label_masks = np.array(label_masks, dtype=np.float32)
    eid2label[eid] = {
      'labels': labels,
      'label_masks': label_masks,
      'gt_eids': gt_eids,
    }

  with open(out_file, 'w') as fout:
    cPickle.dump(eid2label, fout)


if __name__ == '__main__':
  # tst_load_actvid_from_json()
  gen_proposal_label()
