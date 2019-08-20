import os
import json


'''func
'''
def bbx_int_area(p_bbx, v_bbx, format='x1y1x2y2'):
  if p_bbx==[] or v_bbx==[]:
    return 0
  try:
    x = max(p_bbx[0], v_bbx[0])
  except:
    return 0
  y = max(p_bbx[1], v_bbx[1])

  if format == 'x1y1x2y2':
    w = min(p_bbx[2], v_bbx[2]) - x
    h = min(p_bbx[3], v_bbx[3]) - y
  elif format == 'xywh':
    w = min(p_bbx[2] + p_bbx[0], v_bbx[2] + v_bbx[0]) - x
    h = min(p_bbx[3] + p_bbx[1], v_bbx[3] + v_bbx[1]) - y

  if w <= 0 or h <= 0:
    return 0
  else:
    return w*hformat


# x1y1x2y2 or xywh
def bbx_union_area(p_bbx, v_bbx, format='x1y1x2y2'):
  if p_bbx==[] or v_bbx==[]:
    return 0

  if format == 'x1y1x2y2':
    a1 = (p_bbx[2] - p_bbx[0]) * (p_bbx[3] - p_bbx[1])
    a2 = (v_bbx[2] - v_bbx[0]) * (v_bbx[3] - v_bbx[1])
  else:
    a1 = p_bbx[2] * p_bbx[3]
    a2 = p_bbx[2] * p_bbx[3]

  return a1 + a2 - bbx_int_area(p_bbx, v_bbx)


'''expr
'''
def check_videos():
  root_dir = '/home/chenj/data/meva_train'
  video_lst_files = [
    os.path.join(root_dir, 'gt_proposals', 'teamA.lst')
    os.path.join(root_dir, 'gt_proposals', 'teamB.lst')
    os.path.join(root_dir, 'gt_proposals', 'teamC.lst')
  ]
  gen_proposal_dirs = [
    os.path.join(root_dir, 'meva_aug_ind330trn90_feat'),
    os.path.join(root_dir, 'meva_aug_ind330trn72_feat'),
  ]

  videos = set()
  for video_lst_file in video_lst_files:
    with open(video_lst_file) as f:
      for line in f:
        line = line.strip()
        name, _ = os.path.splitext(line)
        videos.add(name)

  missing = []
  for video in videos:
    found = False
    for i in range(2):
      file = os.path.join(gen_proposal_dirs[i], video + '.avi')
      if os.path.exists(file):
        found = True
        break
    if not found:
      missing.append(video)
  print missing


def gen_proposal_label():
  root_dir = '/home/chenj/data/meva_train'
  video_lst = os.path.join(root_dir, 'gt_proposals', 'teamB.lst')
  gt_proposal_dir = os.path.join(root_dir, 'gt_proposals', 'teamB')
  gen_proposal_dir = os.path.join(root_dir, 'meva_aug_ind330trn90_feat')

  videos = []
  with open(video_lst) as f:
    for line in f:
      line = line.strip()
      name, _ = os.path.splitext(line)
      videos.append(name)

  for video in videos:
    gt_file = os.path.join(gt_proposal_dir, video + '.json')
    proposal_file = os.path.join(gen_proposal_dir, video + '.avi', 'annotation', video, 'actv_id_type.pkl')
    pass


if __name__ == '__main__':
  check_videos()
