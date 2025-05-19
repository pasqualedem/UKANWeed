from transformers import SegformerForSemanticSegmentation
import json

import numpy as np
import torch
import torch.nn.functional as F
from torchmetrics.classification import F1Score


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

from Seg_UKAN.dataset import Dataset
from Seg_UKAN.utils import AverageMeter
from albumentations import RandomRotate90,Resize
import time
import torch.nn.functional as F
from transformers import SegformerImageProcessor
from PIL import Image
from torchmetrics.classification import F1Score


INPUT_H = 512
INPUT_W = 512
DATA_DIR = "data/weedmap/test_003"
OUTPUT_DIR = 'outputs'
EXP_NAME = 'segformer'
BATCH_SIZE = 8
NUM_CLASSES = 3
NUM_WORKERS = 4


def f1_multiclass_perClass(output, target, num_class, f1):
    # Convert 3 binary masks into a single-class label map
    if num_class > 1:
      output_class = output
      target_class = torch.argmax(target, dim=1)
      # Compute F1-score per class
      f1_class_scores = f1(output_class, target_class)
      f1.update(output_class, target_class)
    else:
      output_prob = output
      output_pred = output
      # Convert target probabilities to hard labels (0 or 1) using a threshold of 0.5
      target_pred = (target >= 0.5).long()
      f1_class_scores = f1(output_pred, target_pred)
      f1.update(output_class, target_class)
    return f1_class_scores


def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def main(args):
    config = vars(args)
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


    device = config["device"]
    #model = model.cuda()
    model = model.to(device)

    img_ext = '.png'
    mask_ext = '_GroundTruth_color.png'

    # Data loading code
    img_ids = sorted(glob(os.path.join(DATA_DIR, 'images', '*' + img_ext)))
    # img_ids.sort()
    img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]

    #_, val_img_ids = train_test_split(img_ids, test_size=0.2, random_state=config['dataseed'])
    val_img_ids = img_ids

    ckpt = torch.load(f'{OUTPUT_DIR}/{EXP_NAME}/model.pth', map_location=device)

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
        Resize(INPUT_H, INPUT_W),
        transforms.Normalize(),
    ])

    val_dataset = Dataset(
        img_ids=val_img_ids,
        img_dir=os.path.join(DATA_DIR, 'images'),
        mask_dir=os.path.join(DATA_DIR, 'masks'),
        img_ext=img_ext,
        mask_ext=mask_ext,
        num_classes=NUM_CLASSES,
        transform=val_transform)
    print(len(val_dataset))
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        #batch_size=config['batch_size'],
        batch_size = BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        drop_last=False)
    
    f1 = F1Score(num_classes=NUM_CLASSES, task="multiclass" if NUM_CLASSES > 1 else "binary", ignore_index=-100, average=None).to(device)
    image_processor = SegformerImageProcessor(reduce_labels=False)
   # f1_perClass_avg_meter = AverageMeter()
    with torch.no_grad():
        for input, target, meta in tqdm(val_loader, total=len(val_loader)):
            #input = input.cuda()
            #target = target.cuda()
            #model = model.cuda()
            input = input.to(device)
            target = target.to(device)
            model = model.to(device)
            # compute output
            '''
            output_pred = model(pixel_values = input, labels = torch.argmax(target, dim=1))
            loss, output = output_pred.loss, output_pred.logits.cpu()
            '''
            output_pred = model(pixel_values = input)
            output = output_pred.logits.cpu()
            num_classes=output.shape[1]
            # output = outputlogits.numpy()
            dimension_list = [(INPUT_H, INPUT_W) for _ in range(output.shape[0])]

            predicted_segmentation_map = image_processor.post_process_semantic_segmentation(output_pred, target_sizes=dimension_list)
            predicted_segmentation_map = torch.stack(predicted_segmentation_map, dim=0)
            print(predicted_segmentation_map.shape)
            '''
            iou, dice, hd95_ = iou_score(output, target)
            iou_avg_meter.update(iou, input.size(0))
            dice_avg_meter.update(dice, input.size(0))
            hd95_avg_meter.update(hd95_, input.size(0))
            '''
            f1_perClass = f1_multiclass_perClass(predicted_segmentation_map, target, NUM_CLASSES, f1)
           # f1_micro_avg_meter.update(f1_micro, input.size(0))
           # f1_perClass_avg_meter.update(f1_perClass, input.size(0))
            '''
            output = torch.sigmoid(output).cpu().numpy()
            output[output>=0.5]=1
            output[output<0.5]=0
            '''
            # Example: output shape (batch_size, num_classes, height, width)
            # Convert to class indices
            output = predicted_segmentation_map  # Shape: (batch_size, height, width)
            # Convert to one-hot encoding (binary mask for each class)
            output = F.one_hot(output, num_classes=num_classes)  # Shape: (batch_size, height, width, num_classes)
            # Move the class dimension to match original shape (batch_size, num_classes, height, width)
            output = output.permute(0, 3, 1, 2).cpu().numpy()

            for class_idx in range(output.shape[1]):
              os.makedirs(os.path.join(OUTPUT_DIR, EXP_NAME, 'out_val', '{}'.format(class_idx)), exist_ok=True)

            '''
            os.makedirs(os.path.join(args.output_dir, config['name'], 'out_val', '{}'.format(class_idx)), exist_ok=True)
            for pred, img_id in zip(output, meta['img_id']):
                pred_np = pred[0].astype(np.uint8)
                pred_np = pred_np * 255
                img = Image.fromarray(pred_np, 'L')
                img.save(os.path.join(args.output_dir, config['name'], 'out_val/{}.jpg'.format(img_id)))
            '''

            for pred, img_id in zip(output, meta['img_id']):
              for class_idx in range(pred.shape[0]):
                pred_np = pred[class_idx].astype(np.uint8)
                pred_np = pred_np * 255
                img = Image.fromarray(pred_np, 'L')
                img.save(os.path.join(OUTPUT_DIR, EXP_NAME, 'out_val', '{}'.format(class_idx), '{}.jpg'.format(img_id)))


    print(EXP_NAME)

    f1_per_class = f1.compute().tolist()
    f1_macro = f1.compute().mean().item()
    print(f"F1 per class: {f1_per_class}")
    print(f"F1 macro: {f1_macro}")
    
    with open(os.path.join(OUTPUT_DIR, EXP_NAME, 'result.json'), 'w') as f:
        json.dump({
            'f1_per_class': f1_per_class,
            'f1_macro': f1_macro,
        }, f, indent=4)
    
