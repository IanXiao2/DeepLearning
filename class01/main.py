# -*- coding: utf-8 -*-
# @Time    : 2020/6/5 12:04 下午
# @Author  : Ian
# @File    : main.py
# @Project : class01

import torch
import os
import time

from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

import models
import utils
from config import config
from dataset import TextDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(41)
torch.cuda.manual_seed(41)


def train_epoch(model, optimizer, criterion, train_dataloader, show_interval=100):
    model.train()
    acc_meter, loss_meter, iter_count = 0, 0, 0
    for inputs, targets in train_dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        loss_meter += loss.item()
        acc = utils.calc_acc(targets, torch.sigmoid(outputs))
        acc_meter += acc
        iter_count += 1

        if iter_count != 0 and iter_count % show_interval == 0:
            print("train --- %d,loss : %.3e, f1 : %.3f" % (iter_count, loss.item(), acc))

    return loss_meter / iter_count, acc_meter / iter_count


def val_epoch(model, criterion, val_dataloader, threshold=0.5):
    model.eval()
    acc_meter, loss_meter, iter_count = 0, 0, 0
    with torch.no_grad():
        for inputs, targets in val_dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss_meter += loss.item()
            acc = utils.calc_acc(targets, torch.sigmoid(outputs))
            acc_meter += acc
            iter_count += 1
    return loss_meter / iter_count, acc_meter / iter_count


def train():
    model = getattr(models, config.model_name)()
    model = model.to(device)

    train_dataset = TextDataset(train=True)
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=6)
    val_dataset = TextDataset(train=False)
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, num_workers=4)
    print('train size {}, val size {}'.format(len(train_dataset), len(val_dataset)))

    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    scheduler_lr = optim.lr_scheduler.MultiStepLR(optimizer=optimizer,
                                                  milestones=config.stage_epoch,
                                                  gamma=config.lr_decay)
    criterion = nn.BCEWithLogitsLoss()

    model_save_dir = '%s/%s_%s' % (config.ckpt, config.model_name, time.strftime("%Y%m%d%H%M"))
    writer = SummaryWriter(log_dir=model_save_dir, filename_suffix=".IanMac")
    start_epoch = -1
    for epoch in range(start_epoch+1, config.max_epoch):
        since = time.time()
        train_loss, train_acc = train_epoch(model, optimizer, criterion, train_dataloader, show_interval=100)
        val_loss, val_acc = val_epoch(model, criterion, val_dataloader)
        print('#epoch: %02d ---> train loss: %.3e  train f1: %.3f  val loss: %.3e val f1: %.3f time: %s\n'
              % (epoch, train_loss, train_acc, val_loss, val_acc, utils.print_time_cost(since)))
        writer.add_scalars("Loss", {'Train': train_loss}, epoch)
        writer.add_scalars("Loss", {'Valid': val_loss}, epoch)
        writer.add_scalars("ACC", {'Train': train_acc}, epoch)
        writer.add_scalars("ACC", {'Valid': val_acc}, epoch)

        scheduler_lr.step()


if __name__ == '__main__':
    train()
