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
from torchmetrics import F1Score
import yaml

from albumentations.augmentations import transforms
from albumentations.augmentations import geometric

from albumentations.core.composition import Compose, OneOf
from sklearn.model_selection import train_test_split
from torch.optim import lr_scheduler
from tqdm import tqdm
from albumentations import RandomRotate90, Resize

import archs

import losses
from dataset import Dataset

from metrics import f1_multiclass_perClass

from utils import AverageMeter, str2bool

from tensorboardX import SummaryWriter

import shutil
import os
import subprocess

from pdb import set_trace as st


# ARCH_NAMES = archs.__all__
LOSS_NAMES = losses.__all__
LOSS_NAMES.append('BCEWithLogitsLoss')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def list_type(s):
    str_list = s.split(',')
    int_list = [int(a) for a in str_list]
    return int_list


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=None,
                        help='model name: (default: arch+timestamp)')
    parser.add_argument('--epochs', default=400, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch_size', default=8, type=int,
                        metavar='N', help='mini-batch size (default: 16)')

    parser.add_argument('--dataseed', default=2981, type=int,
                        help='')
    
    # model
    parser.add_argument('--arch', '-a', metavar='ARCH', default='UKAN')
    
    parser.add_argument('--deep_supervision', default=False, type=str2bool)
    parser.add_argument('--input_channels', default=3, type=int,
                        help='input channels')
    parser.add_argument('--num_classes', default=1, type=int,
                        help='number of classes')
    parser.add_argument('--input_w', default=256, type=int,
                        help='image width')
    parser.add_argument('--input_h', default=256, type=int,
                        help='image height')
    parser.add_argument('--input_list', type=list_type, default=[128, 160, 256])

    # loss
    parser.add_argument('--loss', default='BCEDiceLoss',
                        choices=LOSS_NAMES,
                        help='loss: ' +
                        ' | '.join(LOSS_NAMES) +
                        ' (default: BCEDiceLoss)')
    parser.add_argument('--loss_weights', default=False, action='store_true',
                        help='apply loss weight')
    
    # dataset
    parser.add_argument('--dataset', default='busi', help='dataset name')      
    parser.add_argument('--data_dir', default='inputs', help='dataset dir')

    parser.add_argument('--output_dir', default='outputs', help='ouput dir')


    # optimizer
    parser.add_argument('--optimizer', default='Adam',
                        choices=['Adam', 'SGD'],
                        help='loss: ' +
                        ' | '.join(['Adam', 'SGD']) +
                        ' (default: Adam)')

    parser.add_argument('--lr', '--learning_rate', default=1e-4, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--weight_decay', default=1e-4, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', default=False, type=str2bool,
                        help='nesterov')

    parser.add_argument('--kan_lr', default=1e-2, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--kan_weight_decay', default=1e-4, type=float,
                        help='weight decay')

    # scheduler
    parser.add_argument('--scheduler', default='CosineAnnealingLR',
                        choices=['CosineAnnealingLR', 'ReduceLROnPlateau', 'MultiStepLR', 'ConstantLR'])
    parser.add_argument('--min_lr', default=1e-5, type=float,
                        help='minimum learning rate')
    parser.add_argument('--factor', default=0.1, type=float)
    parser.add_argument('--patience', default=2, type=int)
    parser.add_argument('--milestones', default='1,2', type=str)
    parser.add_argument('--gamma', default=2/3, type=float)
    parser.add_argument('--early_stopping', default=-1, type=int,
                        metavar='N', help='early stopping (default: -1)')
    parser.add_argument('--cfg', type=str, metavar="FILE", help='path to config file', )
    parser.add_argument('--num_workers', default=4, type=int)

    parser.add_argument('--no_kan', action='store_true')



    config = parser.parse_args()

    return config


def train(config, train_loader, model, criterion, optimizer):
    
    avg_meters = {'loss': AverageMeter()}
    
    model.train()

    pbar = tqdm(total=len(train_loader))
    for input, target, _ in train_loader:
        #input = input.cuda()
        #target = target.cuda()
        input = input.to(device)
        target = target.to(device)

        # compute output
        if config['deep_supervision']:
            outputs = model(input)
            loss = 0
            for output in outputs:
                loss += criterion(output, target)
            loss /= len(outputs)
            
        else:
            output = model(input)
            loss = criterion(output, target)

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

def validate(config, val_loader, model, criterion):
    avg_meters = {'loss': AverageMeter()}

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        pbar = tqdm(total=len(val_loader))
        f1 = F1Score(num_classes=config['num_classes'], task="multiclass" if config['num_classes'] > 1 else "binary", ignore_index=-100, average=None).to(device)
        for input, target, _ in val_loader:
            input = input.to(device)
            target = target.to(device)

            # compute output
            if config['deep_supervision']:
                outputs = model(input)
                loss = 0
                for output in outputs:
                    loss += criterion(output, target)
                loss /= len(outputs)
            else:
                output = model(input)
                loss = criterion(output, target)
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


def main():
    seed_torch()
    config = vars(parse_args())

    exp_name = config.get('name')
    output_dir = config.get('output_dir')

    my_writer = SummaryWriter(f'{output_dir}/{exp_name}')

    if config['name'] is None:
        if config['deep_supervision']:
            config['name'] = '%s_%s_wDS' % (config['dataset'], config['arch'])
        else:
            config['name'] = '%s_%s_woDS' % (config['dataset'], config['arch'])
    
    os.makedirs(f'{output_dir}/{exp_name}', exist_ok=True)

    print('-' * 20)
    for key in config:
        print('%s: %s' % (key, config[key]))
    print('-' * 20)

    with open(f'{output_dir}/{exp_name}/config.yml', 'w') as f:
        yaml.dump(config, f)

    criterion = losses.__dict__[config["loss"]]().to(device)

    cudnn.benchmark = True

    # create model
    model = archs.__dict__[config['arch']](config['num_classes'], config['input_channels'], config['deep_supervision'], embed_dims=config['input_list'], no_kan=config['no_kan'])

    #model = model.cuda()
    model = model.to(device)


    param_groups = []

    kan_fc_params = []
    other_params = []

    for name, param in model.named_parameters():
        # print(name, "=>", param.shape)
        if 'layer' in name.lower() and 'fc' in name.lower(): # higher lr for kan layers
            # kan_fc_params.append(name)
            param_groups.append({'params': param, 'lr': config['kan_lr'], 'weight_decay': config['kan_weight_decay']}) 
        else:
            # other_params.append(name)
            param_groups.append({'params': param, 'lr': config['lr'], 'weight_decay': config['weight_decay']})  
    
    # st()
    if config['optimizer'] == 'Adam':
        optimizer = optim.Adam(param_groups)
    elif config['optimizer'] == 'SGD':
        optimizer = optim.SGD(params, lr=config['lr'], momentum=config['momentum'], nesterov=config['nesterov'], weight_decay=config['weight_decay'])
    else:
        raise NotImplementedError

    if config['scheduler'] == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config['epochs'], eta_min=config['min_lr'])
    elif config['scheduler'] == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=config['factor'], patience=config['patience'], verbose=1, min_lr=config['min_lr'])
    elif config['scheduler'] == 'MultiStepLR':
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[int(e) for e in config['milestones'].split(',')], gamma=config['gamma'])
    elif config['scheduler'] == 'ConstantLR':
        scheduler = None
    else:
        raise NotImplementedError

    #shutil.copy2('/kaggle/input/myukan7/UKAN/Seg_UKAN/train.py', f'{output_dir}/{exp_name}/')
    #shutil.copy2('/kaggle/input/myukan7/UKAN/Seg_UKAN/archs.py', f'{output_dir}/{exp_name}/')

    dataset_name = config['dataset']
    img_ext = '.png'

    if dataset_name == 'busi':
        mask_ext = '_mask.png'
    elif dataset_name == 'glas':
        mask_ext = '.png'
    elif dataset_name == 'cvc':
        mask_ext = '.png'
    elif dataset_name == 'roweeder':
        mask_ext = '_GroundTruth_color.png'

    data_dir = os.path.join(config['data_dir'], "train")
    val_data_dir = data_dir

    # Data loading code
    img_ids = sorted(glob(os.path.join(data_dir, 'images', '*' + img_ext)))
    img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]

    val_img_ids = sorted(glob(os.path.join(val_data_dir, 'images', '*' + img_ext)))
    val_img_ids = [os.path.splitext(os.path.basename(p))[0] for p in val_img_ids]
    
    train_img_ids, val_img_ids = train_test_split(img_ids, test_size=0.2, random_state=config['dataseed'])

    train_transform = Compose([
        RandomRotate90(),
        # geometric.transforms.Flip(),
        Resize(config['input_h'], config['input_w']),
        transforms.Normalize(),
    ])

    val_transform = Compose([
        Resize(config['input_h'], config['input_w']),
        transforms.Normalize(),
    ])

    train_dataset = Dataset(
        img_ids=train_img_ids,
        img_dir=os.path.join(data_dir,  'images'),
        mask_dir=os.path.join(data_dir, 'masks'),
        img_ext=img_ext,
        mask_ext=mask_ext,
        num_classes=config['num_classes'],
        transform=train_transform)

    val_dataset = Dataset(
        img_ids=val_img_ids,
        img_dir=os.path.join(val_data_dir ,'images'),
        mask_dir=os.path.join(val_data_dir,  'masks'),
        img_ext=img_ext,
        mask_ext=mask_ext,
        num_classes=config['num_classes'],
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

    trigger = 0
    best_f1 = 0
    for epoch in range(config['epochs']):
        print('Epoch [%d/%d]' % (epoch, config['epochs']))

        # train for one epoch
        train_log = train(config, train_loader, model, criterion, optimizer)
        # evaluate on validation set
        val_log = validate(config, val_loader, model, criterion)

        if config['scheduler'] == 'CosineAnnealingLR':
            scheduler.step()
        # elif config['scheduler'] == 'ReduceLROnPlateau':
            # scheduler.step(val_log['loss'])

        print('loss %.4f'% (train_log['loss']))

        log['epoch'].append(epoch)
        log['lr'].append(config['lr'])
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
if __name__ == '__main__':
    main()
