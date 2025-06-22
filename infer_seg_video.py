from src.img_seg.model import JepaSegmentation
from src.img_seg.utils import (
    draw_segmentation_map, 
    image_overlay,
    get_segment_labels
)

import argparse
import cv2
import time
import os
import torch
import yaml

parser = argparse.ArgumentParser()
parser.add_argument(
    '--input',
    help='path to the input video',
    default='input/inference_data/videos/video_1.mp4'
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

out_dir = 'outputs/inference_results_video'
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

cap = cv2.VideoCapture(args.input)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
vid_fps = int(cap.get(5))
save_name = args.input.split(os.path.sep)[-1].split('.')[0]

# Define codec and create VideoWriter object.
# out = cv2.VideoWriter(f"{out_dir}/{save_name}.mp4", 
#                     cv2.VideoWriter_fourcc(*'mp4v'), vid_fps, 
#                     (frame_width, frame_height))
out = cv2.VideoWriter(f"{out_dir}/{save_name}.mp4", 
                    cv2.VideoWriter_fourcc(*'mp4v'), vid_fps, 
                    (args.imgsz[1], args.imgsz[0]))

frame_count = 0
total_fps = 0
while cap.isOpened:
    ret, frame = cap.read()
    if ret:
        frame_count += 1
        image = frame
        if args.imgsz is not None:
            image = cv2.resize(image, (args.imgsz[0], args.imgsz[1]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
        # Get labels.
        start_time = time.time()
        # labels = get_segment_labels(image, model, args.device, args.imgsz)
        labels = get_segment_labels(image, model, args.device, image.shape[0:2])
        end_time = time.time()

        fps = 1 / (end_time - start_time)
        total_fps += fps
        
        # Get segmentation map.
        seg_map = draw_segmentation_map(labels.cpu(), viz_map=VIZ_MAP)
        outputs = image_overlay(image, seg_map)
        cv2.putText(
            outputs,
            f"{fps:.1f} FPS",
            (15, 35),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1,
            color=(0, 0, 255),
            thickness=2,
            lineType=cv2.LINE_AA
        )
        out.write(outputs)
        # cv2.imshow('Image', outputs)
        # Press `q` to exit
        # if cv2.waitKey(1) & 0xFF == ord('q'):
            # break
    else:
        break

# Release VideoCapture().
cap.release()
# Close all frames and video windows.
cv2.destroyAllWindows()

# Calculate and print the average FPS.
avg_fps = total_fps / frame_count
print(f"Average FPS: {avg_fps:.3f}")