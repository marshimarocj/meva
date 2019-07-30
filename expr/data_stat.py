import os
import json

import numpy as np


'''func
'''
event_class = {
  "Abandon_Package": 0, 
  "Hand_Interaction": 1,
  "Object_Transfer": 2,
  "People_Talking": 3,
  "Person_Closes_Facility_Door": 4,
  "Person_Closes_Trunk": 5,
  "Person_Closes_Vehicle_Door": 6,
  "Person_Enters_Through_Structure": 7,
  "Person_Enters_Vehicle": 8,
  "Person_Exits_Through_Structure": 9,
  "Person_Exits_Vehicle": 10,
  "Person_Heavy_Carry": 11,
  "Person_Loads_Vehicle": 12,
  "Person_Opens_Facility_Door": 13,
  "Person_Opens_Trunk": 14,
  "Person_Opens_Vehicle_Door": 15,
  "Person-Person_Embrace": 16,
  "Person_Picks_Up_Object": 17,
  "Person_Purchasing": 18,
  "Person_Reading_Document": 19,
  "Person_Sets_Down_Object": 20,
  "Person_Sitting_Down": 21,
  "Person_Standing_Up": 22,
  "Person_Talking_on_Phone": 23,
  "Person_Texting_on_Phone": 24,
  "Person_Unloads_Vehicle": 25,
  "Riding": 26,
  "Vehicle_Drops_Off_Person": 27,
  "Vehicle_Picks_Up_Person": 28,
  "Vehicle_Reversing": 29,
  "Vehicle_Starting": 30,
  "Vehicle_Stopping": 31,
  "Vehicle_Turning_Left": 32,
  "Vehicle_Turning_Right": 33,
  "Vehicle_U-Turn": 34
}

EVENTS = [
  "Abandon_Package", 
  "Hand_Interaction",
  "Object_Transfer",
  "People_Talking",
  "Person_Closes_Facility_Door",
  "Person_Closes_Trunk",
  "Person_Closes_Vehicle_Door",
  "Person_Enters_Through_Structure",
  "Person_Enters_Vehicle",
  "Person_Exits_Through_Structure",
  "Person_Exits_Vehicle",
  "Person_Heavy_Carry",
  "Person_Loads_Vehicle",
  "Person_Opens_Facility_Door",
  "Person_Opens_Trunk",
  "Person_Opens_Vehicle_Door",
  "Person-Person_Embrace",
  "Person_Picks_Up_Object",
  "Person_Purchasing",
  "Person_Reading_Document",
  "Person_Sets_Down_Object",
  "Person_Sitting_Down",
  "Person_Standing_Up",
  "Person_Talking_on_Phone",
  "Person_Texting_on_Phone",
  "Person_Unloads_Vehicle",
  "Riding",
  "Vehicle_Drops_Off_Person",
  "Vehicle_Picks_Up_Person",
  "Vehicle_Reversing",
  "Vehicle_Starting",
  "Vehicle_Stopping",
  "Vehicle_Turning_Left",
  "Vehicle_Turning_Right",
  "Vehicle_U-Turn",
]


'''expr
'''
def event_duration_stat():
  root_dir = '/home/jiac/data/meva' # earth
  label_dirs = [
    os.path.join(root_dir, 'f330', 'train', 'teamA'),
    os.path.join(root_dir, 'f330', 'train', 'teamB'),
    os.path.join(root_dir, 'f330', 'train', 'teamC'),
  ]
  num_class = 35
  out_file = os.path.join(root_dir, 'duration.txt')

  durations = [[] for _ in range(num_class)]
  for label_dir in label_dirs:
    names = os.listdir(label_dir)
    for name in names:
      label_file = os.path.join(label_dir, name)
      with open(label_file) as f:
        data = json.load(f)
      for eid in data:
        proposal = data[eid]
        start_frame = proposal['start_frame']
        end_frame = proposal['end_frame']
        duration = end_frame - start_frame
        event = proposal['event_type']
        if event not in event_class:
          continue
        lid = event_class[event]
        durations[lid].append(duration)

  with open(out_file, 'w') as fout:
    for l in range(num_class):
      # print np.percentile(durations[l], 10), np.percentile(durations[l], 90)
      lower = np.percentile(durations[l], 10)
      upper = np.percentile(durations[l], 90)
      fout.write('%s\t%f\t%f\n'%(EVENTS[l], lower, upper))


if __name__ == '__main__':
  event_duration_stat()
