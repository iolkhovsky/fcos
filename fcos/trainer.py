from common.torch_utils import *
import datetime
import gc
import itertools
import numpy as np
import os
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import traceback
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import torchvision

from dataset.visualization import visualize_batch
from dataset.loader import disbatch
from common.torch_utils import get_available_device
from common.interval import IntervalManager


class FcosTrainer:
    def __init__(self,
                 model,
                 optimizer,
                 train_dataset,
                 epochs,
                 encoder=None,
                 scheduler=None,
                 val_dataset=None,
                 autosave_period=None,
                 validation_period=None,
                 logs_path=None,
                 checkpoints_path=None,
                 grad_clip=None):
        self.model = model
        self.optimizer = optimizer
        self.train_dataset = train_dataset
        self.epochs = epochs
        self.encoder = encoder
        self.scheduler = scheduler
        self.val_dataset = val_dataset
        self.autosave_period = autosave_period
        self.val_period = validation_period
        self.logs_root = logs_path
        self.checkpoints_root = checkpoints_path
        self.grad_clip = grad_clip

        self.device = get_available_device()
        self.metrics_evaluator = MeanAveragePrecision(
            box_format='xyxy',
            iou_type='bbox',
            iou_thresholds=None,
            rec_thresholds=None,
            max_detection_thresholds=None,
            class_metrics=False,
        )
        self.writer = None
        self.checkpoints_path = None
        self.autosave_manager = None
        self.valid_manager = None

    @staticmethod
    def visualize_prediction(images, predictions, target_boxes,
                             target_labels, labels_codec, writer, step, threshold=0.1):
        batch_size = len(images)
        filtered_boxes, filtered_scores, filtered_labels = [], [], []
        for img_idx in range(batch_size):
            mask = predictions['scores'][img_idx] > threshold
            filtered_boxes.append(
                predictions['boxes'][img_idx][mask]
            )
            filtered_scores.append(
                predictions['scores'][img_idx][mask]
            )
            filtered_labels.append(
                predictions['classes'][img_idx][mask]
            )    

        vis_pred_imgs = visualize_batch(
            imgs_batch=images,
            boxes_batch=filtered_boxes,
            labels_batch=filtered_labels,
            scores_batch=filtered_scores,
            codec=labels_codec,
            return_images=True
        )

        vis_target_imgs = visualize_batch(
            imgs_batch=images,
            boxes_batch=target_boxes,
            labels_batch=target_labels,
            codec=labels_codec,
            return_images=True
        )

        pred_images_tensors = [torch.permute(torch.from_numpy(x), (2, 0, 1)) for x in vis_pred_imgs]
        pred_grid = torchvision.utils.make_grid(pred_images_tensors)
        writer.add_image(f'Prediction', pred_grid, step)

        target_images_tensors = [torch.permute(torch.from_numpy(x), (2, 0, 1)) for x in vis_target_imgs]
        target_grid = torchvision.utils.make_grid(target_images_tensors)
        writer.add_image(f'Target', target_grid, step)

    def make_step(self, imgs, boxes, labels, objects_amount, val_iterator, global_step, epoch_idx):
        self.optimizer.zero_grad()
        self.model.train()

        targets = self.encoder(boxes, labels, objects_amount)
        targets = {k: torch.tensor(v, device=self.device) for k, v in targets.items()}
        imgs = imgs.to(self.device)

        loss = self.model(imgs, targets)
        total_loss = loss['total']

        # TODO Remove this debug catch
        skip = False
        if np.any(np.isnan(loss['centerness'])):
            print(f"Centerness loss is NaN on step {global_step}")
            skip = True
        if np.any(np.isnan(loss['classification'])):
            print(f"Classification loss is NaN on step {global_step}")
            skip = True
        if np.any(np.isnan(loss['regression'])):
            print(f"Regression loss is NaN on step {global_step}")
            skip = True
        if skip:
            print("Inputs:")
            print(f"boxes:\t{boxes}")
            print(f"labels:\t{labels}")
            return np.nan

        total_loss.backward()
        if self.grad_clip:
            with torch.autograd.detect_anomaly():
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.optimizer.step()
        total_loss = tensor2numpy(total_loss)

        self.writer.add_scalar(f'Train/TotalLoss', total_loss, global_step)
        self.writer.add_scalar(f'Train/ClassLoss', loss['classification'], global_step)
        self.writer.add_scalar(f'Train/CenterLoss', loss['centerness'], global_step)
        self.writer.add_scalar(f'Train/RegressionLoss', loss['regression'], global_step)

        if self.autosave_manager.check(global_step, epoch_idx):
            model_path = os.path.join(self.checkpoints_path, f"fcos_ep_{epoch_idx}_step_{global_step}")
            self.model.save(model_path)
            print(f"Model state dict has been saved to {model_path}")

        if self.valid_manager.check(global_step, epoch_idx):
            self.model.eval()

            val_imgs, val_boxes, val_labels, val_objects_cnt = next(val_iterator)
            val_batch_size = len(val_imgs)
            val_targets = self.encoder(val_boxes, val_labels, val_objects_cnt)
            val_targets = {k: torch.tensor(v, device=self.device) for k, v in val_targets.items()}
            val_imgs = val_imgs.to(self.device)

            preprocessed_imgs, val_scales = self.model._preprocessor(val_imgs)
            core_outputs = self.model._core(preprocessed_imgs)
            val_loss = self.model._loss(pred=core_outputs, target=val_targets)
            val_total_loss = tensor2numpy(val_loss['total'])

            self.writer.add_scalar(f'Val/TotalLoss', val_total_loss, global_step)
            self.writer.add_scalar(f'Val/ClassLoss', val_loss['classification'], global_step)
            self.writer.add_scalar(f'Val/CenterLoss', val_loss['centerness'], global_step)
            self.writer.add_scalar(f'Val/RegressionLoss', val_loss['regression'], global_step)

            val_boxes, val_labels = disbatch(val_boxes, val_labels, val_objects_cnt)
            val_threshold = 0.05
            val_predictions = self.model._postprocessor(core_outputs, val_scales)
            metrics_pred, metrics_target = [], []
            for img_idx in range(val_batch_size):
                mask = val_predictions['scores'][img_idx] > val_threshold
                metrics_pred.append(
                    {
                        'scores': tensor2cpu(val_predictions['scores'][img_idx][mask]),
                        'boxes': tensor2cpu(val_predictions['boxes'][img_idx][mask]),
                        'labels': tensor2cpu(val_predictions['classes'][img_idx][mask]),
                    }
                )
                metrics_target.append(
                    {
                        'boxes': tensor2cpu(val_boxes[img_idx]),
                        'labels': tensor2cpu(val_labels[img_idx]),
                    }
                )
            self.metrics_evaluator.update(metrics_pred, metrics_target)
            metrics = self.metrics_evaluator.compute()
            self.writer.add_scalar(f'Metrics/mAP', metrics['map'], global_step)
            self.writer.add_scalar(f'Metrics/mAP@50', metrics['map_50'], global_step)
            self.writer.add_scalar(f'Metrics/mAP@75', metrics['map_75'], global_step)
            self.writer.add_scalar(f'Metrics/mAP-small', metrics['map_small'], global_step)
            self.writer.add_scalar(f'Metrics/mAP-medium', metrics['map_medium'], global_step)
            self.writer.add_scalar(f'Metrics/mAP-large', metrics['map_large'], global_step)

            FcosTrainer.visualize_prediction(
                images=val_imgs,
                predictions=val_predictions,
                target_boxes=val_boxes,
                target_labels=val_labels,
                labels_codec=self.model._labels,
                writer=self.writer,
                step=global_step,
                threshold=0.1,
            )

        gc.collect()
        return total_loss

    def run(self):
        session_timestamp = str(datetime.datetime.now())
        session_timestamp = session_timestamp.replace(" ", "").replace(":", "-").replace(".", "-")
        logs_path = os.path.join(
            self.logs_root,
            session_timestamp,
        )
        os.makedirs(logs_path)
        self.checkpoints_path = os.path.join(
            self.checkpoints_root,
            session_timestamp,
        )
        os.makedirs(self.checkpoints_path)

        self.writer = SummaryWriter(logs_path)

        val_iterator = itertools.cycle(iter(self.val_dataset))
        img_batch, _, _, _ = next(iter(self.train_dataset))
        total_batches = len(self.train_dataset)
        total_steps = total_batches * len(img_batch)

        self.model = self.model.train().to(self.device)

        self.autosave_manager = IntervalManager(self.autosave_period)
        self.valid_manager = IntervalManager(self.val_period)

        global_step = 0
        with tqdm(total=total_steps) as pbar:
            for epoch_idx in range(self.epochs):
                for step, (imgs, boxes, labels, objects_amount) in enumerate(self.train_dataset):
                    try:
                        total_loss = self.make_step(imgs, boxes, labels, objects_amount, val_iterator, global_step, epoch_idx)
                    except Exception as e:
                        print(f"Error: Got an unhandled exception during epoch {epoch_idx} step {step}")
                        print(traceback.format_exc())

                    pbar.set_description(
                        f"Epoch: {epoch_idx}/{self.epochs} "
                        f"Step {step}/{total_steps} Loss: {total_loss}"
                    )
                    pbar.update(1)
                    global_step += 1
