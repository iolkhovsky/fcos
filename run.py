import cv2
import numpy as np
import torch

from fcos import FCOS, build_backbone
from common.torch_utils import get_available_device
from dataset import VocLabelsCodec
from dataset.visualization import visualize_batch

checkpoint = "/Users/iolkhovsky/Documents/repos/fcos/checkpoints/2023-02-0521-42-16-544953/fcos_ep_0_step_260"
model = FCOS.load(
    path=checkpoint,
    backbone=build_backbone('resnet50'),
    labels_codec=VocLabelsCodec(),
    res=(256, 256),
)
device = get_available_device()
model = model.to(device)
cap = cv2.VideoCapture(0)


def visualize(img, pred, threshold=0.1):
    filtered_boxes, filtered_scores, filtered_labels = [], [], []
    for img_idx in range(1):
        mask = pred['scores'][img_idx] > threshold
        filtered_boxes.append(
            pred['boxes'][img_idx][mask]
        )
        filtered_scores.append(
            pred['scores'][img_idx][mask]
        )
        filtered_labels.append(
            pred['classes'][img_idx][mask]
        )    

    vis_pred_imgs = visualize_batch(
        imgs_batch=img,
        boxes_batch=filtered_boxes,
        labels_batch=filtered_labels,
        scores_batch=filtered_scores,
        codec=VocLabelsCodec(),
        return_images=True
    )
    
    return vis_pred_imgs[0]

while True:
    ret, frame = cap.read()
    img_tensor = torch.tensor([cv2.resize(frame, (256, 256))]).to(device)

    detections = model.forward(img_tensor)
    vis_img = visualize(img_tensor, detections, 0.2)
    cv2.imshow('vis', vis_img)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()