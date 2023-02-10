import datetime
import itertools
import os
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import traceback
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import torchvision

from dataset.visualization import visualize_batch
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


    def run(self):
        session_timestamp = str(datetime.datetime.now())
        session_timestamp = session_timestamp.replace(" ", "").replace(":", "-").replace(".", "-")
        logs_path = os.path.join(
            self.logs_root,
            session_timestamp,
        )
        os.makedirs(logs_path)
        checkpoints_path = os.path.join(
            self.checkpoints_root,
            session_timestamp,
        )
        os.makedirs(checkpoints_path)

        writer = SummaryWriter(logs_path)

        val_iterator = itertools.cycle(iter(self.val_dataset))
        img_batch, _, _ = next(iter(self.train_dataset))
        total_batches = len(self.train_dataset)
        total_steps = total_batches * len(img_batch)

        self.model = self.model.train().to(self.device)

        autosave_manager = IntervalManager(self.autosave_period)
        valid_manager = IntervalManager(self.val_period)

        global_step = 0
        with tqdm(total=total_steps) as pbar:
            for epoch_idx in range(self.epochs):
                for step, (imgs, boxes, labels) in enumerate(self.train_dataset):
                    try:
                        self.optimizer.zero_grad()

                        targets = self.encoder(boxes, labels)
                        imgs = imgs.to(self.device)
                        loss = self.model(imgs, targets)

                        total_loss = loss['classification'] + loss['centerness'] + loss['regression']
                        skip = False

                        if torch.any(torch.isnan(loss['centerness'])):
                            print(f"Centerness loss is NaN on step {global_step}")
                            skip = True
                        if torch.any(torch.isnan(loss['classification'])):
                            print(f"Classification loss is NaN on step {global_step}")
                            skip = True
                        if torch.any(torch.isnan(loss['regression'])):
                            print(f"Regression loss is NaN on step {global_step}")
                            skip = True

                        if skip:
                            print("Inputs:")
                            print(f"boxes:\t{boxes}")
                            print(f"labels:\t{labels}")
                            continue

                        total_loss.backward()
                        if self.grad_clip:
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                        self.optimizer.step()

                        writer.add_scalar(f'Train/TotalLoss', total_loss.detach().numpy(), global_step)
                        writer.add_scalar(f'Train/ClassLoss', loss['classification'].detach().numpy(), global_step)
                        writer.add_scalar(f'Train/CenterLoss', loss['centerness'].detach().numpy(), global_step)
                        writer.add_scalar(f'Train/RegressionLoss', loss['regression'].detach().numpy(), global_step)

                        if autosave_manager.check(global_step, epoch_idx):
                            model_path = os.path.join(checkpoints_path, f"fcos_ep_{epoch_idx}_step_{step}")
                            self.model.save(model_path)
                            print(f"Model state dict has been saved to {model_path}")

                        if valid_manager.check(global_step, epoch_idx):
                            self.model.eval()

                            val_imgs, val_boxes, val_labels = next(val_iterator)
                            val_batch_size = len(val_imgs)
                            val_targets = self.encoder(val_boxes, val_labels)
                            val_imgs = val_imgs.to(self.device)

                            preprocessed_imgs, val_scales = self.model._preprocessor(val_imgs)
                            core_outputs = self.model._core(preprocessed_imgs)
                            val_loss = self.model._loss(pred=core_outputs, target=val_targets)
                            val_total_loss = val_loss['classification'] + val_loss['centerness'] + val_loss['regression']

                            writer.add_scalar(f'Val/TotalLoss', val_total_loss.detach().numpy(), global_step)
                            writer.add_scalar(f'Val/ClassLoss', val_loss['classification'].detach().numpy(), global_step)
                            writer.add_scalar(f'Val/CenterLoss', val_loss['centerness'].detach().numpy(), global_step)
                            writer.add_scalar(f'Val/RegressionLoss', val_loss['regression'].detach().numpy(), global_step)

                            val_threshold = 0.05
                            val_predictions = self.model._postprocessor(core_outputs, val_scales)
                            metrics_pred, metrics_target = [], []
                            for img_idx in range(val_batch_size):
                                mask = val_predictions['scores'][img_idx] > val_threshold
                                metrics_pred.append(
                                    {
                                        'scores': val_predictions['scores'][img_idx][mask].to('cpu'),
                                        'boxes': val_predictions['boxes'][img_idx][mask].to('cpu'),
                                        'labels': val_predictions['classes'][img_idx][mask].to('cpu'),
                                    }
                                )
                                metrics_target.append(
                                    {
                                        'boxes': val_boxes[img_idx].to('cpu'),
                                        'labels': val_labels[img_idx].to('cpu'),
                                    }
                                )
                            self.metrics_evaluator.update(metrics_pred, metrics_target)
                            metrics = self.metrics_evaluator.compute()
                            writer.add_scalar(f'Metrics/mAP', metrics['map'], global_step)
                            writer.add_scalar(f'Metrics/mAP@50', metrics['map_50'], global_step)
                            writer.add_scalar(f'Metrics/mAP@75', metrics['map_75'], global_step)
                            writer.add_scalar(f'Metrics/mAP-small', metrics['map_small'], global_step)
                            writer.add_scalar(f'Metrics/mAP-medium', metrics['map_medium'], global_step)
                            writer.add_scalar(f'Metrics/mAP-large', metrics['map_large'], global_step)

                            FcosTrainer.visualize_prediction(
                                images=val_imgs,
                                predictions=val_predictions,
                                target_boxes=val_boxes,
                                target_labels=val_labels,
                                labels_codec=self.model._labels,
                                writer=writer,
                                step=global_step,
                                threshold=0.1,
                            )

                            self.model.train()
                    except Exception as e:
                        print(f"Error: Got an unhandled exception during epoch {epoch_idx} step {step}")
                        print(traceback.format_exc())

                    pbar.set_description(
                        f"Epoch: {epoch_idx}/{self.epochs} "
                        f"Step {step}/{total_steps} Loss: {total_loss.detach().cpu().numpy()}"
                    )
                    pbar.update(1)
                    global_step += 1
