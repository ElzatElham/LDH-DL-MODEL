# -*- coding: utf-8 -*-
"""
Created on Fri May 24 14:08:11 2024

@author: 13621
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 17:07:48 2024

@author: 13621
"""
from ultralytics import YOLO
import argparse


parser = argparse.ArgumentParser(description='Train a YOLOv8 model')
parser.add_argument("--weights",    type=str, help="initial weights path")
parser.add_argument('--data',       type=str, help='dataset.yaml path')
parser.add_argument("--device",     default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")




args    = parser.parse_args()
model   = args.weights
datadir = args.data



model = YOLO(model)
# print(model_dir)





metrics = model.val(data=datadir, device   = args.device)

metrics.box.map    # map50-95
metrics.box.map50  # map50
metrics.box.map75  # map75
metrics.box.maps   # a list contains map50-95 of each category

# Map50 = float(f"{metrics.box.map50*100:.1f}")
# Map75 = float(f"{metrics.box.map75*100:.1f}")
# Map   = float(f"{metrics.box.map*100:.1f}")
# print(metrics.box.map50*100,metrics.box.map75*100,metrics.box.map*100)
# print(Map50,Map75,Map)
# print(metrics.box)






















