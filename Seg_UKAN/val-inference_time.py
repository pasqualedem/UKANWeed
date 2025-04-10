#! /data/cxli/miniconda3/envs/th200/bin/python
import argparse
import os
from glob import glob
import random
import numpy as np

import cv2
import torch
import torch.backends.cudnn as cudnn
import yaml
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from collections import OrderedDict

import archs

from dataset import Dataset
from metrics import iou_score
from utils import AverageMeter
from albumentations import RandomRotate90,Resize
from accelerate import Accelerator
import time
import statistics

from PIL import Image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
WARM_UP = 10
ITERATIONS = 2
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=None, help='model name')
    parser.add_argument('--output_dir', default='outputs', help='ouput dir')
            
    args = parser.parse_args()

    return args

def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def main():
    seed_torch()
    args = parse_args()

    with open(f'{args.output_dir}/{args.name}/config.yml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    print('-'*20)
    for key in config.keys():
        print('%s: %s' % (key, str(config[key])))
    print('-'*20)

    cudnn.benchmark = True

    model = archs.__dict__[config['arch']](config['num_classes'], config['input_channels'], config['deep_supervision'], embed_dims=config['input_list'])

    #model = model.cuda()
    model = model.to(device)

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

    # Data loading code
    img_ids = sorted(glob(os.path.join(config['data_dir'], 'images', '*' + img_ext)))
    # img_ids.sort()
    img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]

    #_, val_img_ids = train_test_split(img_ids, test_size=0.2, random_state=config['dataseed'])
    val_img_ids = img_ids
    ckpt = torch.load(f'{args.output_dir}/{args.name}/model.pth', map_location=device)

    try:        
        model.load_state_dict(ckpt)
    except:
        print("Pretrained model keys:", ckpt.keys())
        print("Current model keys:", model.state_dict().keys())

        pretrained_dict = {k: v for k, v in ckpt.items() if k in model.state_dict()}
        current_dict = model.state_dict()
        diff_keys = set(current_dict.keys()) - set(pretrained_dict.keys())

        print("Difference in model keys:")
        for key in diff_keys:
            print(f"Key: {key}")

        model.load_state_dict(ckpt, strict=False)
        
    model.eval()

    val_transform = Compose([
        Resize(config['input_h'], config['input_w']),
        transforms.Normalize(),
    ])

    val_dataset = Dataset(
        img_ids=val_img_ids,
        img_dir=os.path.join(config['data_dir'], 'images'),
        mask_dir=os.path.join(config['data_dir'], 'masks'),
        img_ext=img_ext,
        mask_ext=mask_ext,
        num_classes=config['num_classes'],
        transform=val_transform)
    print(len(val_dataset))
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        #batch_size = len(val_dataset),
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False)
    i = 0
    total_time = 0
    with torch.no_grad():
        '''
        for input, target, meta in tqdm(val_loader, total=len(val_loader)):
            #input = input.cuda()
            #target = target.cuda()
            #model = model.cuda()
            input = input.to(device)
            target = target.to(device)
            model = model.to(device)
            # Warm-up runs
            for _ in range(WARM_UP):
                _ = model(input)
            # Measure inference time with GPU synchronization
            # Multiple iterations
            num_iterations = ITERATIONS
            inference_times = []
            if device.type == 'cuda':
              for _ in range(num_iterations):
                torch.cuda.synchronize()
                start_time = time.time()
                output = model(input)
                torch.cuda.synchronize()
                end_time = time.time()
                inference_times.append(end_time - start_time)
            else:
              for _ in range(num_iterations):
                start_time = time.time()
                output = model(input)
                end_time = time.time()
                inference_times.append(end_time - start_time)
                
            average_inference_time = statistics.mean(inference_times)*1000
            print(f"Average inference time: {average_inference_time:.0f} ms")
        '''
        accelerator = Accelerator()
        for k in range(ITERATIONS):
          print(f"Iteration {k}")
          for input, target, meta in tqdm(val_loader, total=len(val_loader)):
              input = input.to(device)
              #target = target.to(device)
              model = model.to(device)
              if i < WARM_UP:
                _ = model(input)
                continue
          start_time = time.time()
          _ = model(input)
          end_time = time.time()
          # Wait for GPU to finish
          if device.type == 'cuda':
            torch.cuda.synchronize()
          # Calculate elapsed time
          elapsed_time = end_time - start_time
          total_time += elapsed_time
          i += 1
    accelerator.wait_for_everyone()
    average_time = total_time / i
    print(f"Average time: {average_time* 1000:.0f} ms")
if __name__ == '__main__':
    main()
