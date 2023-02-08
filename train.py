import argparse
import os
import torch

from common.utils import *
from common.interval import Interval
import dataset
from fcos import FCOS, build_backbone, FcosDetectionsEncoder, FcosTrainer


def parse_cmd_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=os.path.join("configs", "train.yaml"),
                        help="Path to training config")
    args = parser.parse_args()
    return args


def compile_model(model_config):
    return FCOS(
         backbone=build_backbone(model_config['backbone']),
         labels_codec=getattr(dataset, model_config['labels'])(),
         res=tuple(model_config['resolution'])
    )


def compile_datasets(dataset_config):
    root = dataset_config["root"]
    download = bool(dataset_config["download"])
    train_dataloader = dataset.build_dataloader(
        subset='train',
        batch_size=dataset_config['train']['batch_size'],
        root=root,
        download=download
    )
    val_dataloader = dataset.build_dataloader(
        subset='val',
        batch_size=dataset_config['val']['batch_size'],
        root=root,
        download=download
    )
    return train_dataloader, val_dataloader


def compile_optimizer(model_pars, optimizer_config):
    optimizer_type = optimizer_config['type']
    optimizer_pars = optimizer_config['parameters']
    optimizer_pars['params'] = model_pars
    optimizer_pars['lr'] = float(optimizer_pars['lr'])
    optimizer = getattr(torch.optim, optimizer_type)(**optimizer_pars)
    return optimizer


def compile_scheduler(optimizer, scheduler_config):
    scheduler_type = scheduler_config['type']
    scheduler_pars = scheduler_config['parameters']
    scheduler_pars['optimizer'] = optimizer
    scheduler = getattr(torch.optim.lr_scheduler, scheduler_type)(**scheduler_pars)
    return scheduler


def run_training(args):
    config = read_yaml(args.config)
    pretty_print(config)

    train_dataset, val_dataloader = compile_datasets(config['dataset'])
    model_config = config['model']
    labels_codec = getattr(dataset, model_config['labels'])()
    encoder = FcosDetectionsEncoder(
        res=model_config['resolution'],
        labels=labels_codec,
    )

    model = compile_model(model_config)
    optimizer = compile_optimizer(model.parameters(), config['optimizer'])
    scheduler = compile_scheduler(optimizer, config['scheduler'])
    grad_clip = float(config['gradient_clip'])

    epochs = config['epochs']
    autosave_period = Interval.from_config(config['autosave_period'])
    validation_period = Interval.from_config(config['validation_period'])

    logs_path = config['logs']['path']
    checkpoints_path = config['checkpoints']['path']

    trainer = FcosTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_dataset=train_dataset,
        val_dataset=val_dataloader,
        encoder=encoder,
        epochs=epochs,
        autosave_period=autosave_period,
        validation_period=validation_period,
        logs_path=logs_path,
        checkpoints_path=checkpoints_path,
        grad_clip=grad_clip,
    )
    trainer.run()

if __name__ == "__main__":
    run_training(parse_cmd_args())
