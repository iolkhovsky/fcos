import argparse
import cv2
import os
import time
import torch

from common.utils import pretty_print, read_yaml
import dataset
from dataset.visualization import visualize_batch
from common.torch_utils import get_available_device
from fcos import FCOS, build_backbone


def parse_cmd_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=os.path.join("configs", "run.yaml"),
                        help="Path to training config")
    args = parser.parse_args()
    return args


def compile_model(model_config, device=None):
    model = FCOS.load(
        path=model_config['checkpoint'],
        backbone=build_backbone(model_config['backbone']),
        labels_codec=getattr(dataset, model_config['labels'])(),
        res=tuple(model_config['resolution'])
    )
    if device: 
        model = model.to(device)
    return model


def img2tensor(image, res, device=None):
    tensor = torch.tensor([cv2.resize(image, res)])
    if device:
       tensor = tensor.to(device)
    return tensor


def visualize(img, pred, labels_codec, threshold=0.1):
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
        codec=labels_codec,
        return_images=True
    )
    
    return vis_pred_imgs[0]


class Profiler:
    def __init__(self, hint):
        assert isinstance(hint, str)
        self._hint = hint
        self._start = None

    def __enter__(self):
        self._start = time.time()

    def __exit__(self, type, value, traceback):
        duration = time.time() - self._start
        duration_s = "{:.3f}".format(duration)
        fps =  "{:.2f}".format(1. / (duration + 1e-4))
        print(f"{self._hint} duration (sec): {duration_s}\tFPS={fps}")


def run(args):
    config = read_yaml(args.config)
    pretty_print(config)

    device = get_available_device()
    model = compile_model(config['model'], device)

    video_source = str(config['stream']['device'])
    if video_source.isdigit():
        video_source = int(video_source)
    resolution = tuple(config['model']['resolution'])
    threshold = float(config['threshold'])
    labels_codec = getattr(dataset, config['model']['labels'])()
    cap = cv2.VideoCapture(video_source)

    profiler = Profiler("Inference")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        img_tensor = img2tensor(frame, resolution, device)
        with profiler:
            detections = model.forward(img_tensor)
        vis_img = visualize(img_tensor, detections, labels_codec, threshold)
        cv2.imshow('Prediction', vis_img)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run(parse_cmd_args())
