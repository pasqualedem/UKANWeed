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
from ptflops import get_model_complexity_info

import time

from PIL import Image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {n_params}")

    macs, params = get_model_complexity_info(
          model,
          (3, 512, 512),
          as_strings=True,
          backend="pytorch",
          print_per_layer_stat=False,
          verbose=False,
    )
    print(f"MACs: {macs}")
    print(f"Params: {params}")
 



if __name__ == '__main__':
    main()
