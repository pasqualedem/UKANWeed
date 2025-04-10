import numpy as np
import torch
import torch.nn.functional as F
from torchmetrics.classification import F1Score

from medpy.metric.binary import jc, dc, hd, hd95, recall, specificity, precision



def iou_score(output, target):
    smooth = 1e-5

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    output_ = output > 0.5
    target_ = target > 0.5
    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()
    iou = (intersection + smooth) / (union + smooth)
    dice = (2* iou) / (iou+1)

    try:
        hd95_ = hd95(output_, target_)
    except:
        hd95_ = 0
    
    return iou, dice, hd95_


def dice_coef(output, target):
    smooth = 1e-5

    output = torch.sigmoid(output).view(-1).data.cpu().numpy()
    target = target.view(-1).data.cpu().numpy()
    intersection = (output * target).sum()

    return (2. * intersection + smooth) / \
        (output.sum() + target.sum() + smooth)

def f1_multiclass_perClass(output, target, f1):
    num_class = output.shape[1]
    # Convert 3 binary masks into a single-class label map
    if num_class > 1:
      output_class = torch.argmax(output, dim=1)
      target_class = torch.argmax(target, dim=1)
      # Compute F1-score per class
      f1_class_scores = f1(output_class, target_class)
      f1.update(output_class, target_class)
    else:
      output_prob = torch.sigmoid(output)
      output_pred = (output_prob >= 0.5).long()
      # Convert target probabilities to hard labels (0 or 1) using a threshold of 0.5
      target_pred = (target >= 0.5).long()
      f1_class_scores = f1(output_pred, target_pred)
      f1.update(output_class, target_class)
    return f1_class_scores

def f1_multiclass_micro(output, target):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_class = output.shape[1]
    f1 = F1Score(num_classes=num_class, task="multiclass" if num_class > 1 else "binary", ignore_index=-100, average='micro').to(device)
    if num_class > 1:
      output_class = torch.argmax(output, dim=1)
      target_class = torch.argmax(target, dim=1)
      # Compute F1-score per class
      f1_score = f1(output_class, target_class)
    else:
      output_prob = torch.sigmoid(output)
      output_pred = (output_prob >= 0.5).long()
      # Convert target probabilities to hard labels (0 or 1) using a threshold of 0.5
      target_pred = (target >= 0.5).long()
      f1_score = f1(output_pred, target_pred)
    return f1_score

def indicators(output, target):
    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    output_ = output > 0.5
    target_ = target > 0.5

    iou_ = jc(output_, target_)
    dice_ = dc(output_, target_)
    hd_ = hd(output_, target_)
    hd95_ = hd95(output_, target_)
    recall_ = recall(output_, target_)
    specificity_ = specificity(output_, target_)
    precision_ = precision(output_, target_)

    return iou_, dice_, hd_, hd95_, recall_, specificity_, precision_
