import argparse
import os
import torch

from common.utils import *
import dataset
from fcos import FCOS, build_backbone, FcosDetectionsEncoder


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
    train_dataloader = dataset.build_dataloader(
        'train', batch_size=dataset_config['train']['batch_size'])
    val_dataloader = dataset.build_dataloader(
        'val', batch_size=dataset_config['val']['batch_size'])
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

    model_config = config['model']
    model = compile_model(model_config)

    labels_codec = getattr(dataset, model_config['labels'])()
    encoder = FcosDetectionsEncoder(
        res=model_config['resolution'],
        labels=labels_codec,
    )

    train_dataset, val_dataloader = compile_datasets(config['dataset'])
    optimizer = compile_optimizer(model.parameters(), config['optimizer'])
    scheduler = compile_scheduler(optimizer, config['scheduler'])


if __name__ == "__main__":
    run_training(parse_cmd_args())
