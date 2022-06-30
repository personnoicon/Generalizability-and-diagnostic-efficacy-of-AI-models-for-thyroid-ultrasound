# coding: utf-8
# *****************************************************
# Aitrox Information Technology
# http://www.proxima-ai.com
# Copyright 2021 Aitrox. All rights reserved.
# CreateDate: 2021-08-26
# ******************************************************

import os
import torch
import tqdm
import numpy as np
import torchvision
from time import strftime, localtime
from torchtoolbox.transform import Cutout
from warmup_scheduler import GradualWarmupScheduler

from src.models import DensNet121
from src.dataloader import ThyroidDataset
from src.focalLoss import FocalLoss
from src.utils import calculate_metrics_binary_classification

from torchvision import transforms
from torch.nn.modules.loss import CrossEntropyLoss
from torch.optim import Adam, SGD
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter

# torch.cuda.set_device(4)
device = torch.device('cuda')
# torch.backends.cudnn.benchmark = True


num_workers = 10
lr = 0.001
input_size = 512
batch_size = 32
save_model_freq = 10
total_epoch = 200
warn_up_epoch = 20
mean = [0, 0, 0]
std = [1, 1, 1]
# mean=[0.485, 0.456, 0.406]
# std=[0.229, 0.224, 0.225]
weight_decay = 5e-4
momentum = 0.9

weight_save_dir = os.path.join('backup', str(strftime('%Y-%m-%d-%H-%M', localtime())))
if not os.path.exists(weight_save_dir):
    os.makedirs(weight_save_dir)


def save_model(model, save_path, epoch):
    """
    :param model: model
    :param save_path: model save path
    :param epoch: epoch
    :return:
    """
    model_file = os.path.join(save_path, 'checkpoint_{}.pth'.format(str(epoch)))
    if 'module' in dir(model):
        torch.save(model.module.state_dict(), model_file)
    else:
        torch.save(model.state_dict(), model_file)
    print('---saved pth file:{}---'.format(model_file))


def data_proprecess(img_dir):
    """
    :param img_dir: dataimgs
    :return:  Dataloder
    """
    train_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.RandomRotation(degrees=80),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        # Cutout(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    val_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    val_annos = os.path.join(img_dir, 'val.json')
    train_annos = os.path.join(img_dir, 'train.json')
    val_dataset = ThyroidDataset(val_annos, val_transform, input_size=input_size)
    train_dataset = ThyroidDataset(train_annos, train_transform, input_size=input_size)
    train_dataloder = DataLoader(train_dataset,
                                 batch_size=batch_size,
                                 num_workers=num_workers,
                                 shuffle=True)

    val_dataloder = DataLoader(val_dataset,
                               batch_size=batch_size,
                               num_workers=num_workers,
                               shuffle=True)

    one_epoch_batches = int(np.ceil(len(train_dataset) / batch_size))

    return train_dataloder, val_dataloder, one_epoch_batches


model = DensNet121(2)  # len(train_dataset.classes())
model.train()
model = model.to(device)

optimizer = SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, momentum=momentum,
                weight_decay=weight_decay)
# optimizer = Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15960, gamma=0.1) # 1470
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9, last_epoch=-1)
scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=1.0, total_epoch=warn_up_epoch,
                                          after_scheduler=scheduler)

if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model, device_ids=[0, 1])

criterion = CrossEntropyLoss()
# criterion = FocalLoss()
logger = SummaryWriter(log_dir=weight_save_dir)


if __name__ == '__main__':
    best_f1 = 0
    min_val_loss = 2.0
    epoch = 0
    iteration = 0
    train_dataloder, val_dataloder, one_epoch_batches = data_proprecess(img_dir='labels')
    while 1:
        valid_losses = []
        batch_losses = []
        valid_sum_loss = 0.0
        for i, (imgs, targets) in enumerate(tqdm.tqdm(train_dataloder)):
            imgs, targets = imgs.to(device), targets.to(device)
            targets = targets.squeeze(1)
            optimizer.zero_grad()
            result = model(imgs)
            loss = criterion(result, targets.type(torch.long))
            batch_loss = loss.item()
            loss.backward()
            optimizer.step()
            scheduler_warmup.step(epoch)
            # scheduler.step()

            if i % one_epoch_batches == 0:
                torchvision.utils.save_image(imgs, os.path.join(weight_save_dir, 'train_batch_imgs.jpg'),
                                             padding=0, normalize=True)

            logger.add_scalar('lr', optimizer.state_dict()['param_groups'][0]['lr'], iteration)
            logger.add_scalar('train_loss', batch_loss, iteration)
            batch_losses.append(batch_loss)
            result_train = torch.max(result, 1)[1]

            with torch.no_grad():
                result = calculate_metrics_binary_classification(result_train.cpu().numpy(), targets.cpu().numpy())
                for metric in result:
                    logger.add_scalar('train/' + metric, result[metric], iteration)

            if iteration % one_epoch_batches == 0:
                model.eval()
                with torch.no_grad():
                    model_result = []
                    val_targets = []
                    for i, (imgs, batch_targets) in enumerate(tqdm.tqdm(val_dataloder)):
                        imgs = imgs.to(device)
                        val_batch_targets = batch_targets.to(device)
                        val_batch_targets = val_batch_targets.squeeze(1)
                        model_batch_result = model(imgs)
                        val_loss = criterion(model_batch_result, val_batch_targets.type(torch.long))
                        valid_loss = val_loss.item()
                        valid_losses.append(valid_loss)
                        model_batch_val_result = torch.max(model_batch_result, 1)[1]
                        model_result.extend(model_batch_val_result.cpu().numpy())
                        val_targets.extend(batch_targets.cpu().numpy())

                result = calculate_metrics_binary_classification(np.array(model_result), np.array(val_targets))
                for metric in result:
                    logger.add_scalar('test/' + metric, result[metric], iteration)
                print("epoch:{:2d} test: "
                      "micro f1: {:.3f} "
                      "macro f1: {:.3f} "
                      "current lr:{} ".format(epoch,
                                              result['micro/f1'],
                                              result['macro/f1'],
                                              optimizer.state_dict()['param_groups'][0]['lr']))

                avg_val_loss = np.mean(valid_losses)
                print("--val loss:{:.3f}--".format(avg_val_loss))
                logger.add_scalar('val_loss', avg_val_loss, epoch)
                model.train()

                if avg_val_loss < min_val_loss:
                    try:
                        model_file = os.path.join(weight_save_dir, 'model_best.pth')
                        torch.save(model.state_dict(), model_file)
                        min_val_loss = avg_val_loss
                        print('--saved best model sucessed, val_loss:{:.4f}--'.format(min_val_loss))
                    except:
                        pass

            iteration += 1
        loss_value = np.mean(batch_losses)
        print("---epoch:{:2d} iter:{:3d} train loss:{:.3f}---\n".format(epoch, iteration, loss_value))
        if epoch % save_model_freq == 0:
            if epoch > 0:
                save_model(model=model, save_path=weight_save_dir, epoch=epoch)
        epoch += 1
        if total_epoch < epoch:
            break
