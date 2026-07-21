import argparse

import cv2
import numpy as np
import torch
from torch import optim
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Tuple, Callable

from datasets.mnist_dataset import visualise_sample
from datasets import DatasetFactory
from models import ModelFactory
from torchmetrics.classification import MulticlassAccuracy

from yaml import load, Loader

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True, help='path to config file')
    args = vars(ap.parse_args())
    return args


def train_cycle(model: nn.Module, device: torch.device, train_loader: DataLoader, test_dataloader: DataLoader,
                criterion: nn.Module, accuracy: nn.Module | None, optimizer: optim.Optimizer,
                epoch: int, viz_iter: int, viz_func: Callable | None = None) -> Tuple[nn.Module, float, float]:
    model.train()
    train_losses = []
    test_losses = []

    train_accuracy = []
    test_accuracy = []
    dataloader_size = len(train_loader)
    for batch_idx, batch in enumerate(train_loader):
        for key in batch:
            batch[key] = batch[key].to(device)
        label = batch.pop('label')
        output = model(batch)
        loss = criterion(output, label)
        if accuracy is not None:
            with torch.no_grad():
                acc = accuracy(output, label)
                train_accuracy.append(acc.item() * 100)
        train_losses.append(loss.item())
        show_line = f'TRAIN epoch: {epoch}, batch: {batch_idx}/{dataloader_size}, loss: {round(loss.item(), 3)}'
        if accuracy is not None:
            show_line += f', acc: {round(acc.item() * 100, 2)}%'
        print(show_line)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if viz_func is not None:
            if batch_idx % viz_iter == 0:
                with torch.no_grad():
                    vis_sample = visualise_sample(batch, output)
                cv2.imshow('sample', vis_sample)
                cv2.waitKey(10)

    mean_train_loss = np.mean(train_losses)
    print("Train epoch: ", epoch, 'Loss', np.round(mean_train_loss, 3))
    if len(train_accuracy) > 0:
        print("Train accuracy: ", np.round(np.mean(train_accuracy), 3))

    # test
    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_dataloader):
            for key in batch:
                batch[key] = batch[key].to(device)
            label = batch.pop('label')
            output = model(batch)
            loss = criterion(output, label)
            test_losses.append(loss.item())
            if accuracy is not None:
                acc = accuracy(output, label)
                test_accuracy.append(acc.item() * 100)
            print(f'TEST epoch: {epoch}, batch: {batch_idx}, loss: {round(loss.item(), 3)}, acc: {round(acc.item() * 100, 2)}%')

    mean_test_loss = np.mean(test_losses)
    print("Test epoch: ", epoch, 'Loss', np.round(mean_test_loss, 3))
    if len(test_accuracy) > 0:
        print("Test accuracy: ", np.round(np.mean(test_accuracy), 3))

    return model, mean_train_loss, mean_test_loss

def get_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    return device

def create_dataset(dataset_cfg):
    dataset_class = DatasetFactory.get_dataset_type(dataset_cfg)
    dataset_params = dataset_cfg.get('params')
    train_dataset = dataset_class(**dataset_params, is_train=True)
    test_dataset = dataset_class(**dataset_params, is_train=False)
    return train_dataset, test_dataset

def create_optimizer(optimizer_cfg, model_params) -> optim.Optimizer:
    optimizer_type = optimizer_cfg.pop('type')
    if optimizer_type == 'Adam':
        optimizer = optim.Adam(model_params, **optimizer_cfg)
    elif optimizer_type == 'SGD':
        optimizer = optim.SGD(model_params, **optimizer_cfg)
    else:
        raise ValueError

    return optimizer

def create_loss(loss_cfg: dict) -> nn.Module:
    loss_type = loss_cfg.pop('type')
    if loss_type == 'CrossEntropyLoss':
        loss = nn.CrossEntropyLoss()
    else:
        raise ValueError
    return loss

def create_accuracy(accuracy_cfg: dict) -> nn.Module | None:
    if accuracy_cfg is None:
        return None
    accuracy_type = accuracy_cfg.pop('type')
    if accuracy_type == 'MulticlassAccuracy':
        return MulticlassAccuracy(**accuracy_cfg)
    else:
        raise ValueError

def main(config_path: str):
    cfg = load(open(config_path, 'r'), Loader=Loader)
    model = ModelFactory.get_model(cfg.get('model'))

    train_cfg = cfg.get('train')
    batch_size = train_cfg.get('batch_size')
    epoch_number = train_cfg.get('epochs')
    assert epoch_number is not None
    viz_iteration = train_cfg.get('viz_iteration', 50)
    num_workers = train_cfg.get('num_workers', 1)

    device = get_device()
    print("Device: ", device)
    model = model.to(device)

    train_dataset, test_dataset = create_dataset(cfg.get('dataset'))
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    viz_func = train_dataset.get_vis_sample_func()

    optimizer = create_optimizer(cfg.get('optimizer'), model.parameters())
    loss = create_loss(cfg.get('loss'))
    accuracy = create_accuracy(cfg.get('accuracy'))
    if accuracy is not None:
        accuracy = accuracy.to(device)
    for i in range(epoch_number):
        model, train_loss, test_loss = train_cycle(model, device, train_dataloader, test_dataloader,
                                                   loss, accuracy, optimizer, i, viz_iteration, viz_func=viz_func)
        # TODO: implement log writer


if __name__ == '__main__':
    args = parse_args()
    main(args['config'])