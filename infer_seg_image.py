"""
Script to run image inference.

USAGE:
python infer_seg_image.py --input input/inference_data/ \
--model outputs/img_seg/best_model_iou.pth \
--config segmentation_configs/voc.yaml
"""

from src.img_seg.model import JepaSegmentation
from src.img_seg.utils import (
    draw_segmentation_map, 
    image_overlay,
    get_segment_labels
)

import argparse
import cv2
import os
import glob
import torch
import yaml

parser = argparse.ArgumentParser()
parser.add_argument(
    '--input',
    help='path to the input image directory',
    default='input/inference_data/images'
)
parser.add_argument(
    '--device',
    default='cuda:0',
    help='compute device, cpu or cuda'
)
parser.add_argument(
    '--imgsz', 
    default=[448, 448],
    type=int,
    nargs='+',
    help='width, height'
)
parser.add_argument(
    '--model',
    default='outputs/model_iou.pth'
)
parser.add_argument(
    '--config',
    required=True,
    help='path to the dataset configuration file'
)
args = parser.parse_args()

out_dir = 'outputs/inference_results_image'
os.makedirs(out_dir, exist_ok=True)

# Set configurations.
with open(args.config, 'r') as f:
    config = yaml.safe_load(f)
    f.close()

ALL_CLASSES = config['ALL_CLASSES']
VIZ_MAP = config['VIS_LABEL_MAP']

model = JepaSegmentation(
    fine_tune=False, 
    weights=None,
    num_classes=len(ALL_CLASSES)
)

ckpt = torch.load(args.model)
model.load_state_dict(ckpt['model_state_dict'])
_ = model.to(args.device).eval()

image_paths = glob.glob(os.path.join(args.input, '*'))
for image_path in image_paths:
    image = cv2.imread(image_path)
    if args.imgsz is not None:
        image = cv2.resize(image, (args.imgsz[0], args.imgsz[1]))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Get labels.
    labels = get_segment_labels(image, model, args.device, args.imgsz)
    
    # Get segmentation map.
    seg_map = draw_segmentation_map(labels.cpu(), viz_map=VIZ_MAP)

    outputs = image_overlay(image, seg_map)
    cv2.imshow('Image', outputs)
    cv2.waitKey(0)
    
    # Save path.
    image_name = image_path.split(os.path.sep)[-1]
    save_path = os.path.join(
        out_dir, '_'+image_name
    )
    cv2.imwrite(save_path, outputs)