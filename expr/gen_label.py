import os
import json
import cPickle

import numpy as np

from diva_common.structure.annotation import *
from diva_common.util.tracklet import *


SPATIAL_DELTA = 0.5

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


def calc_spatial_iou(emeta, gt_emeta, start, end):
  int_fr_set = set(range(start, end)) & set(range(gt_emeta.event_begin, gt_emeta.event_end))
  print int_fr_set

  if len(int_fr_set)==0:
    return 0

  s_iou_list=[]
  for fr in int_fr_set:
    if fr not in emeta.frame2bbx or fr not in gt_emeta.frame2bbx:
      continue
    ch_bbx=list(emeta.frame2bbx[fr])
    gt_bbx=list(gt_emeta.frame2bbx[fr])

    try:
      s_iou = bbx_int_area(ch_bbx, gt_bbx) / float(bbx_union_area(ch_bbx,gt_bbx))
    except e:
      s_iou = 0
    print ch_bbx, gt_bbx

    #   calc s_iou for bbx
    s_iou_list.append(s_iou)

  if len(s_iou_list)==0:
    return 0

  return np.mean(s_iou_list)


def gen_proposal_label(label2lid, gt_label_file, prop_file, out_file):
  STRIDE = 8 # frames

  gt_actv = load_actvid_from_json(gt_label_file)
  actv = load_actvid_from_json(prop_file)

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
    gt_eids = list(gt_eids)

    labels = np.array(labels, dtype=np.float32)
    label_masks = np.array(label_masks, dtype=np.float32)
    eid2label[eid] = {
      'labels': labels,
      'gt_eids': gt_eids,
    }

  with open(out_file, 'w') as fout:
    cPickle.dump(eid2label, fout)


'''expr
'''
def tst_load_actvid_from_json():
  root_dir = '/mnt/sda/jiac'
  file = os.path.join(root_dir, 'f330_train_annotation', 'teamB', '2018-03-09_10-15-01_10-20-01_bus_G505.json')

  actv = load_actvid_from_json(file)


def gen_proposal_label_one_video():
  root_dir = '/mnt/sda/jiac'
  video = '2018-03-07_17-20-00_17-25-00_school_G330'
  gt_label_file = os.path.join(root_dir, 'f330_train_annotation', 'teamB', video + '.json')
  prop_file = os.path.join(root_dir, 'f330_train_fb_feat', 'trn', 'indoor', video + '.avi', 'annotation', video, 'actv_id_type.json')
  label_file = os.path.join('/home/chenj/data', 'meva_train', 'label.json')
  out_file = os.path.join('/home/chenj/data/label', video + '.pkl')

  STRIDE = 8 # frames

  gt_actv = load_actvid_from_json(gt_label_file)
  actv = load_actvid_from_json(prop_file)
  print len(gt_actv.eid2event_meta), len(actv.eid2event_meta)

  with open(label_file) as f:
    label2lid = json.load(f)
  num_label = len(label2lid)

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
        print f, siou
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


def gen_split_video_lst():
  root_dir = '/home/chenj/data'
  # lst_file = os.path.join(root_dir, 'lst', 'aws.trn.lst')
  # out_file = os.path.join(root_dir, 'lst', 'trn.lst')
  lst_file = os.path.join(root_dir, 'lst', 'aws.val.lst')
  out_file = os.path.join(root_dir, 'lst', 'val.lst')

  with open(lst_file) as f, open(out_file, 'w') as fout:
    for line in f:
      line = line.strip()
      start = line.rfind('/')+1
      end = -11
      fout.write(line[start:end] + '\n')


def bat_gen_proposal_label():
  root_dir = '/mnt/sda/jiac'
  lst_file = os.path.join('/home/chenj/data', 'lst', 'trn.lst')
  label_file = os.path.join('/home/chenj/data', 'meva_train', 'label.json')

  gt_label_dir = os.path.join(root_dir, 'f330_train_annotation', 'teamB')
  prop_dir = os.path.join(root_dir, 'f330_train_fb_feat', 'trn')

  with open(label_file) as f:
    label2lid = json.load(f)
  num_label = len(label2lid)

  videos = []
  with open(lst_file) as f:
    for line in f:
      video = line.strip()
      videos.append(video)

  for video in videos:
    gt_label_file = os.path.join(gt_label_dir, video + '.json')
    print gt_label_file
    if not os.path.exists(gt_label_file):
      continue

    prop_file = os.path.join(prop_dir, 'indoor', video + '.avi', 'annotation', 'actv_id_type.json')
    print prop_file
    if not os.path.exists(prop_file):
      prop_file = os.path.join(prop_dir, 'outdoor', video + '.avi', 'annotation', 'actv_id_type.json')
      print prop_file
      if not os.path.exists(prop_file):
        continue

    out_file = os.path.join('/home/chenj/data/label', video + '.pkl')

    print video

    gen_proposal_label(label2lid, gt_label_file, prop_file, out_file)


if __name__ == '__main__':
  # tst_load_actvid_from_json()
  # gen_proposal_label_one_video()
  # gen_split_video_lst()
  bat_gen_proposal_label()
