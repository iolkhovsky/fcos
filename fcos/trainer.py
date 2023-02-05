import datetime
import itertools
import os
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import traceback
from torchmetrics.detection.mean_ap import MeanAveragePrecision

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
                        total_loss.backward()
                        if self.grad_clip:
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                        self.optimizer.step()

                        writer.add_scalar(f'Train/TotalLoss', total_loss.detach(), global_step)
                        writer.add_scalar(f'Train/ClassLoss', loss['classification'].detach(), global_step)
                        writer.add_scalar(f'Train/CenterLoss', loss['centerness'].detach(), global_step)
                        writer.add_scalar(f'Train/RegressionLoss', loss['regression'].detach(), global_step)

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

                            writer.add_scalar(f'Val/TotalLoss', val_total_loss.detach(), global_step)
                            writer.add_scalar(f'Val/ClassLoss', val_loss['classification'].detach(), global_step)
                            writer.add_scalar(f'Val/CenterLoss', val_loss['centerness'].detach(), global_step)
                            writer.add_scalar(f'Val/RegressionLoss', val_loss['regression'].detach(), global_step)

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
                            print("\nMetrics:")
                            print(metrics)

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
