import argparse
import collections
import torch.nn as nn
import torch
from ruamel.yaml import YAML
import yaml

class qkv_transform(nn.Conv1d):
    """Conv1d for qkv_transform"""

def str2bool(v):
    if v.lower() in ['true', 1]:
        return True
    elif v.lower() in ['false', 0]:
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def create_rgb_segmentation(segmentation, num_classes=None):
    """
    Convert a segmentation map to an RGB visualization using a precise colormap.

    Args:
        segmentation (torch.Tensor): Segmentation map of shape [B, H, W] where
                                      each pixel contains class labels (natural numbers).
        num_classes (int): The number of unique classes in the segmentation.

    Returns:
        torch.Tensor: RGB visualization of shape [B, 3, H, W].
    """
    if len(segmentation.shape) == 4:
        segmentation = segmentation.argmax(dim=1)
    if num_classes is None:
        num_classes = segmentation.max().item() + 1
    
    # Define a precise colormap for specific classes
    colormap = torch.tensor([
        [0, 0, 0],         # Class 0: Black (Background)
        [0, 128, 0],       # Class 1: Green
        [128, 0, 0],       # Class 2: Red
        [128, 128, 0],     # Class 3: Yellow
        [0, 0, 128],       # Class 4: Blue
        [128, 0, 128],     # Class 5: Magenta
        [0, 128, 128],     # Class 6: Cyan
        [192, 192, 192],   # Class 7: Light Gray
        [255, 0, 0],       # Class 8: Bright Red
        [0, 255, 0],       # Class 9: Bright Green
        [0, 0, 255],       # Class 10: Bright Blue
        [255, 255, 0],     # Class 11: Bright Yellow
        [255, 0, 255],     # Class 12: Bright Magenta
        [0, 255, 255],     # Class 13: Bright Cyan
        [128, 128, 128],   # Class 14: Dark Gray
        [255, 165, 0],     # Class 15: Orange
        [75, 0, 130],      # Class 16: Indigo
        [255, 20, 147],    # Class 17: Deep Pink
        [139, 69, 19],     # Class 18: Brown
        [154, 205, 50],    # Class 19: Yellow-Green
        [70, 130, 180],    # Class 20: Steel Blue
        [220, 20, 60],     # Class 21: Crimson
        [107, 142, 35],    # Class 22: Olive Drab
        [0, 100, 0],       # Class 23: Dark Green
        [205, 133, 63],    # Class 24: Peru
        [148, 0, 211],     # Class 25: Dark Violet
    ], dtype=torch.uint8)  # Ensure dtype is uint8

    # Initialize an empty tensor for RGB output
    B, H, W = segmentation.shape
    rgb_segmentation = torch.zeros((B, 3, H, W), dtype=torch.uint8)

    # Loop through each class and assign the corresponding RGB color
    for class_id in range(num_classes):
        # Create a mask for the current class
        class_mask = (segmentation == class_id).unsqueeze(1)  # Shape: [B, 1, H, W]
        # Assign the corresponding color to the rgb_segmentation
        rgb_segmentation += class_mask * colormap[class_id].view(1, 3, 1, 1)  # Broadcasting

    return rgb_segmentation


def unnormalize_img(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    Unnormalize an image tensor.

    Args:
        img (torch.Tensor): Image tensor of shape [C, H, W].
        mean (list): List of mean values for each channel.
        std (list): List of standard deviation values for each channel.

    Returns:
        torch.Tensor: Unnormalized image tensor.
    """
    mean = torch.tensor(mean, device=img.device).view(1, 3, 1, 1)
    std = torch.tensor(std, device=img.device).view(1, 3, 1, 1)
    img = img * std + mean
    return img


def nested_dict_update(d, u):
    if u is not None:
        if isinstance(d, dict):
            for k, v in u.items():
                if isinstance(v, collections.abc.Mapping):
                    d[k] = nested_dict_update(d.get(k) or {}, v)
                else:
                    d[k] = v
        elif isinstance(d, list):
            d = [u]
    return d


def load_yaml(file_path):
    try:
        yaml = YAML()
        with open(file_path, "r") as yaml_file:
            data = yaml.load(yaml_file.read())
            # data = strip_wandb_keys(data)
            return data
    except FileNotFoundError as e:
        print(f"File '{file_path}' not found.")
        raise e
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
        raise e
        

def write_yaml(data: dict, file_path: str = None, file=None):
    """ Write a dictionary to a YAML file.

    Args:
        data (dict): the data to write
        file_path (str): the path to the file
        file: the file object to write to (esclusive with file_path)
    """
    if file is not None:
        file.write(yaml.dump(data))
        return
    if file_path is None:
        raise ValueError("file_path or file must be specified")
    try:
        with open(file_path, "w") as yaml_file:
            yaml.dump(data, yaml_file)
    except FileNotFoundError:
        print(f"File '{file_path}' not found.")
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
        
        
class PrintLogger:
    def __init__(self, print_fn=print):
        self.print_fn = print_fn

    def log(self, *args, **kwargs):
        self.print_fn(*args, **kwargs)
        
    def info(self, *args, **kwargs):
        self.print_fn(*args, **kwargs)
        
    def warning(self, *args, **kwargs):
        self.print_fn(*args, **kwargs)
        
    def error(self, *args, **kwargs):
        self.print_fn(*args, **kwargs)