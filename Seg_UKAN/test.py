#! /data/cxli/miniconda3/envs/th200/bin/python
import argparse
import json
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
from metrics import iou_score, f1_multiclass_micro, f1_multiclass_perClass
from utils import AverageMeter, create_rgb_segmentation, load_yaml, unnormalize_img
from albumentations import RandomRotate90, Resize
import pandas as pd
import torch.nn.functional as F
from torchmetrics.classification import F1Score
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--name", default=None, help="model name")
    parser.add_argument("--output_dir", default="outputs", help="ouput dir")
    parser.add_argument(
        "--device",
        default="cuda:0",
        type=str,
        help="device to use for training / testing",
    )

    args = parser.parse_args()

    return args


def save_predictions(input, target, output, img_ids, output_dir, config):
    for img, gt, pred, img_id in zip(input, target, output, img_ids):
        unnormalized_img = unnormalize_img(img).squeeze(0)
        unnormalized_img = unnormalized_img.permute(1, 2, 0).cpu().numpy()
        unnormalized_img = (unnormalized_img*255).astype(np.uint8)
        unnormalized_img = Image.fromarray(unnormalized_img, "RGB")
        unnormalized_img.save(
            os.path.join(
                output_dir,
                config["name"],
                "out_val",
                f"{img_id}.png",
            )
        )
        pred = create_rgb_segmentation(
            torch.from_numpy(pred).unsqueeze(0),
            num_classes=config["num_classes"],
        )
        gt = create_rgb_segmentation(
            gt.cpu().unsqueeze(0),
            num_classes=config["num_classes"],
        )
        pred = pred.squeeze(0).permute(1, 2, 0).numpy()
        gt = gt.squeeze(0).permute(1, 2, 0).numpy()
        img_pred = pred.astype(np.uint8)
        img_gt = gt.astype(np.uint8)
        img_pred  = Image.fromarray(img_pred, "RGB")
        img_gt = Image.fromarray(img_gt, "RGB")
        img_pred.save(
            os.path.join(
                output_dir,
                config["name"],
                "out_val",
                f"{img_id}_pred.png",
            )
        )
        img_gt.save(
            os.path.join(
                output_dir,
                config["name"],
                "out_val",
                f"{img_id}_gt.png",
            )
        )


def seed_torch(seed=1029):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def test(output_dir, name, device="cuda"):

    config = load_yaml(f"{output_dir}/{name}/config.yml")

    print("-" * 20)
    for key in config.keys():
        print("%s: %s" % (key, str(config[key])))
    print("-" * 20)

    cudnn.benchmark = True
    
    if isinstance(config["input_list"], str):
        config["input_list"] = [int(i) for i in config["input_list"].split(",")]

    model = archs.__dict__[config["arch"]](
        config["num_classes"],
        config["input_channels"],
        config["deep_supervision"],
        embed_dims=config["input_list"],
        no_kan=config["no_kan"],
    )

    device = torch.device(device)
    model = model.to(device)

    dataset_name = config["dataset"]
    img_ext = ".png"

    if dataset_name == "busi":
        mask_ext = "_mask.png"
    elif dataset_name == "glas":
        mask_ext = ".png"
    elif dataset_name == "cvc":
        mask_ext = ".png"
    elif dataset_name == "roweeder":
        mask_ext = "_GroundTruth_color.png"

    test_data_dir = os.path.join(config["data_dir"], "test")
    # Data loading code
    img_ids = sorted(glob(os.path.join(test_data_dir, "images", "*" + img_ext)))
    # img_ids.sort()
    img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]
    print("img_ids", len(img_ids))

    # _, val_img_ids = train_test_split(img_ids, test_size=0.2, random_state=config['dataseed'])
    val_img_ids = img_ids
    ckpt = torch.load(f"{output_dir}/{name}/model.pth", map_location=device)

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

    val_transform = Compose(
        [
            Resize(config["input_h"], config["input_w"]),
            transforms.Normalize(),
        ]
    )

    test_data_dir = os.path.join(config["data_dir"], "test")
    print(f"test_data_dir: {test_data_dir}")

    val_dataset = Dataset(
        img_ids=val_img_ids,
        img_dir=os.path.join(test_data_dir, "images"),
        mask_dir=os.path.join(test_data_dir, "masks"),
        img_ext=img_ext,
        mask_ext=mask_ext,
        num_classes=config["num_classes"],
        transform=val_transform,
    )
    print(f"Data len", len(val_dataset))
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        # batch_size = len(val_dataset),
        shuffle=False,
        num_workers=config["num_workers"],
        drop_last=False,
    )

    f1_micro_avg_meter = AverageMeter()
    f1 = F1Score(
        num_classes=config["num_classes"],
        task="multiclass" if config["num_classes"] > 1 else "binary",
        ignore_index=-100,
        average=None,
    ).to(device)
    # f1_perClass_avg_meter = AverageMeter()
    os.makedirs(os.path.join(output_dir, config['name'], 'out_val'), exist_ok=True)
    with torch.no_grad():
        for input, target, meta in tqdm(val_loader, total=len(val_loader)):
            # input = input.cuda()
            # target = target.cuda()
            # model = model.cuda()
            input = input.to(device)
            target = target.to(device)
            model = model.to(device)
            # compute output
            output = model(input)

            f1_multiclass_perClass(output, target, f1)

            # Example: output shape (batch_size, num_classes, height, width)
            # Convert to class indices
            num_classes = output.shape[1]
            output = torch.argmax(output, dim=1)  # Shape: (batch_size, height, width)
            # Convert to one-hot encoding (binary mask for each class)
            output = F.one_hot(
                output, num_classes=num_classes
            )  # Shape: (batch_size, height, width, num_classes)
            # Move the class dimension to match original shape (batch_size, num_classes, height, width)
            output = output.permute(0, 3, 1, 2).cpu().numpy()

            save_predictions(input, target, output, meta["img_id"], output_dir, config)

    print(config["name"])

    print("F1 micro: %.4f" % f1_micro_avg_meter.avg)
    print(f"F1 per class: {f1.compute().tolist()}")
    print(f"F1 Macro avg: {f1.compute().mean().item()}")
    f1_per_class = f1.compute().tolist()
    f1_macro = f1.compute().mean().item()
    print(f"F1 per class: {f1_per_class}")
    print(f"F1 macro: {f1_macro}")

    with open(os.path.join(output_dir, name, "result.json"), "w") as f:
        json.dump(
            {
                "f1_per_class": f1_per_class,
                "f1_macro": f1_macro,
            },
            f,
            indent=4,
        )

    if os.path.exists(os.path.join(output_dir, "overall_results.csv")):
        overall_results = pd.read_csv(
            os.path.join(output_dir, "overall_results.csv")
        )
    else:
        overall_results = pd.DataFrame(
            columns=["model", "f1_0", "f1_1", "f1_2", "avg_f1"]
        )
        
    new_row = pd.DataFrame([{
        "model": name,
        "f1_0": f1_per_class[0],
        "f1_1": f1_per_class[1],
        "f1_2": f1_per_class[2],
        "avg_f1": f1_macro,
    }])
    overall_results = pd.concat([overall_results, new_row], ignore_index=True)
    overall_results.to_csv(
        os.path.join(output_dir, "overall_results.csv"), index=False
    )


def main():
    seed_torch()
    args = parse_args()
    test(
        output_dir=args.output_dir,
        name=args.name,
        device=args.device,
    )


if __name__ == "__main__":
    main()
