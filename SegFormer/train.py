from torchmetrics import F1Score
from transformers import SegformerForSemanticSegmentation
import json
import argparse
import os
from collections import OrderedDict
from glob import glob
import random
import albumentations
import numpy as np

import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import yaml

from albumentations.augmentations import transforms
from albumentations.augmentations import geometric

from albumentations.core.composition import Compose, OneOf
from sklearn.model_selection import train_test_split
from torch.optim import lr_scheduler
from tqdm import tqdm
from albumentations import RandomRotate90, Resize
from tensorboardX import SummaryWriter

from Seg_UKAN.dataset import Dataset
from Seg_UKAN.metrics import f1_multiclass_perClass
from Seg_UKAN.utils import AverageMeter, str2bool
from Seg_UKAN.test import test
import Seg_UKAN.losses

import shutil
import os
import subprocess

from pdb import set_trace as st

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
INPUT_H = 512
INPUT_W = 512
DATA_DIR = "data/weedmap/RedEdge_processed"
BATCH_SIZE = 2
NUM_CLASSES = 3
NUM_WORKERS = 4
#SEED = 1029


def list_type(s):
    str_list = s.split(',')
    return [int(a) for a in str_list]


def train(train_loader, model, optimizer, criterion):

    avg_meters = {'loss': AverageMeter()}

    model.train()

    pbar = tqdm(total=len(train_loader))
    for input, target, _ in train_loader:
        #input = input.cuda()
        #target = target.cuda()
        input = input.to(device)
        target = target.to(device)

        labels = torch.argmax(target, dim=1)
        output_pred = model(pixel_values = input).logits
        output_pred = torch.nn.functional.interpolate(output_pred, size=labels.shape[-2:], mode='bilinear', align_corners=False)
        
        loss = criterion(output_pred, labels)

        # compute gradient and do optimizing step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_meters['loss'].update(loss.item(), input.size(0))
        postfix = OrderedDict([
            ('loss', avg_meters['loss'].avg),
        ])
        pbar.set_postfix(postfix)
        pbar.update(1)
    pbar.close()
    return OrderedDict([('loss', avg_meters['loss'].avg)])


def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
      torch.cuda.manual_seed(seed)
      torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    
def validate(config, val_loader, model, criterion):
    avg_meters = {'loss': AverageMeter()}

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        pbar = tqdm(total=len(val_loader))
        f1 = F1Score(num_classes=NUM_CLASSES, task="multiclass" if NUM_CLASSES > 1 else "binary", ignore_index=-100, average=None).to(device)
        for input, target, _ in val_loader:
            input = input.to(device)
            target = target.to(device)
            labels = torch.argmax(target, dim=1)

            # compute output
            if config['deep_supervision']:
                outputs = model(input).logits
                loss = sum(criterion(output, target) for output in outputs)
                loss /= len(outputs)
            else:
                output = model(input).logits
                output = torch.nn.functional.interpolate(output, size=labels.shape[-2:], mode='bilinear', align_corners=False)
                loss = criterion(output, labels)
                f1_multiclass_perClass(output, target, f1)

            avg_meters['loss'].update(loss.item(), input.size(0))
            f1_value = f1.compute().mean().item()

            postfix = OrderedDict([
                ('loss', avg_meters['loss'].avg),
                ('f1', f1_value),
            ])
            pbar.set_postfix(postfix)
            pbar.update(1)
        pbar.close()


    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('f1', f1_value),
                        ])


def main(config):
    config = vars(config)
    loss_weights = config["loss_weights"]
    exp_name = config["name"]
    output_dir = config["output_dir"]
    lr = config["lr"]
    epochs = config["epochs"]

    my_writer = SummaryWriter(f'{output_dir}/{exp_name}')

    # load id2label mapping from a JSON on the hub
    label2id = {'background':0, 'crop':1, 'weed':2}
    id2label = {0:'background', 1:'crop', 2:'weed'}

    # define model
    model = SegformerForSemanticSegmentation.from_pretrained("nvidia/mit-b0",
                                                            num_labels=3,
                                                            id2label=id2label,
                                                            label2id=label2id,
    )

    seed_torch()

    cudnn.benchmark = True
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.00006)
    criterion = Seg_UKAN.losses.__dict__['FocalLoss'](class_weighting=loss_weights).to(device)

    img_ext = '.png'
    mask_ext = '_GroundTruth_color.png'

    data_dir = os.path.join(config['data_dir'], "train")
    val_data_dir = data_dir

    # Data loading code
    img_ids = sorted(glob(os.path.join(data_dir, 'images', f'*{img_ext}')))
    img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]

    train_img_ids, val_img_ids = train_test_split(img_ids, test_size=0.2, random_state=config['dataseed'])

    train_transform = Compose([
        RandomRotate90(),
        # geometric.transforms.Flip(),
        Resize(INPUT_H, INPUT_W),
        transforms.Normalize(),
    ])

    val_transform = Compose([
        Resize(INPUT_H, INPUT_W),
        transforms.Normalize(),
    ])

    train_dataset = Dataset(
        img_ids=train_img_ids,
        img_dir=os.path.join(data_dir,  'images'),
        mask_dir=os.path.join(data_dir, 'masks'),
        img_ext=img_ext,
        mask_ext=mask_ext,
        num_classes=NUM_CLASSES,
        transform=train_transform)

    val_dataset = Dataset(
        img_ids=val_img_ids,
        img_dir=os.path.join(val_data_dir ,'images'),
        mask_dir=os.path.join(val_data_dir,  'masks'),
        img_ext=img_ext,
        mask_ext=mask_ext,
        num_classes=NUM_CLASSES,
        transform=val_transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        drop_last=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False)

    log = OrderedDict([
        ('epoch', []),
        ('lr', []),
        ('loss', []),
        ('val_loss', []),
        ('val_f1', []),
    ])

    best_f1 = 0
    trigger = 0

    os.makedirs(f'{output_dir}/{exp_name}', exist_ok=True)
    for epoch in range(epochs):
        print('Epoch [%d/%d]' % (epoch, epochs))

        # train for one epoch
        train_log = train( train_loader, model, optimizer, criterion)
        # evaluate on validation set
        val_log = validate(config, val_loader, model, criterion)

        print('loss %.4f'% (train_log['loss']))

        log['epoch'].append(epoch)
        log['lr'].append(lr)
        log['loss'].append(train_log['loss'])
        log['val_loss'].append(val_log['loss'])
        log['val_f1'].append(val_log['f1'])

        pd.DataFrame(log).to_csv(f'{output_dir}/{exp_name}/log.csv', index=False)

        my_writer.add_scalar('train/loss', train_log['loss'], global_step=epoch)
        my_writer.add_scalar('val/loss', val_log['loss'], global_step=epoch)
        my_writer.add_scalar('val/f1', val_log['f1'], global_step=epoch)

        my_writer.add_scalar('val/best_f1_value', best_f1, global_step=epoch)

        trigger += 1

        if val_log['f1'] > best_f1:
            torch.save(model.state_dict(), f'{output_dir}/{exp_name}/model.pth')
            best_f1 = val_log['f1']
            print("=> saved best model")
            print('F1: %.4f' % best_f1)
            trigger = 0

        # early stopping
        if config['early_stopping'] >= 0 and trigger >= config['early_stopping']:
            print("=> early stopping")
            break

        if torch.cuda.is_available():
          torch.cuda.empty_cache()
    torch.save(model.state_dict(), f'{output_dir}/{exp_name}/model.pth')