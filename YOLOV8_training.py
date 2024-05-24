# -*- coding: utf-8 -*-
"""
Created on Fri May 24 13:02:24 2024

@author: 13621
"""





from ultralytics import YOLO
from pathlib import Path
import numpy as np
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Train a YOLOv8 model')
parser.add_argument("--weights",    type=str, help="initial weights path")
parser.add_argument('--epochs',     type=int, help='total training epochs')
parser.add_argument('--data',       type=str, help='dataset.yaml path')
parser.add_argument("--batch",      type=int, default=16, help="total batch size for all GPUs, -1 for autobatch")
parser.add_argument("--imgsz",      type=int, default=512, help="train, val image size (pixels)")
parser.add_argument("--resume",     nargs="?", const=True, default=False, help="resume most recent training")
parser.add_argument("--device",     default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
parser.add_argument("--fliplr",     type=int, default=0, help="image flip left-right (probability)")
parser.add_argument("--flipud",     type=int, default=0, help="image flip up-down (probability)")
parser.add_argument("--patience",   type=int, default=100, help="EarlyStopping patience (epochs without improvement)")
parser.add_argument("--name",       default="exp", help="save to project/name")

args    = parser.parse_args()
model   = args.weights
datadir = args.data






# print(model)
# detection_model = YOLO(model)  # build a new model from YAML
# print(model)
results = YOLO(args.weights).train(
    data        = datadir,
    epochs      = args.epochs,
    imgsz       = args.imgsz,
    patience    = args.patience,
    batch       = args.batch,
    fliplr      = args.fliplr,
    flipud      = args.flipud,
    name        = args.name,
    device      = args.device
)















