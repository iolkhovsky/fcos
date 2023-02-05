import datetime
import os
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import traceback

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
                            pass
                    except Exception as e:
                        print(f"Error: Got an unhandled exception during epoch {epoch_idx} step {step}")
                        print(traceback.format_exc())

                    pbar.set_description(
                        f"Epoch: {epoch_idx}/{self.epochs} "
                        f"Step {step}/{total_steps} Loss: {total_loss.detach().cpu().numpy()}"
                    )
                    pbar.update(1)
                    global_step += 1
