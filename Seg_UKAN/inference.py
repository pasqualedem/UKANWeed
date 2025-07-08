#! /data/cxli/miniconda3/envs/th200/bin/python
import argparse
import copy
import os
import random
import numpy as np

import pandas as pd
from ptflops import get_model_complexity_info
import torch
import torch.backends.cudnn as cudnn
from torchtnt.utils.flops import FlopTensorDispatchMode
from tqdm import tqdm
import yaml

from utils import load_yaml
import archs

from transformers import ResNetForImageClassification, AutoImageProcessor

WARM_UP = 10
ITERATIONS = 100
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=None, help='model name')
    parser.add_argument('--output_dir', default='outputs', help='ouput dir')
    parser.add_argument('--device', default='cuda:0', type=str,
                        help='device to use for training / testing')
    parser.add_argument('--batch_size', default=None, type=int,
                        metavar='N', help='mini-batch size (default: 16)')
            
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


class EmbeddedResNet(torch.nn.Module):
    def __init__(self, model: ResNetForImageClassification, processor: AutoImageProcessor, input_size):
        super(EmbeddedResNet, self).__init__()
        self.model = model
        self.processor = processor
        self.input_size = input_size
        
    def forward(self, x):
        return self.model(x)


def get_resnet(size):
    """
    Returns a ResNet model with the specified input size.
    """
    # Load the pre-trained ResNet model
    model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")
    processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")

    return EmbeddedResNet(model, processor, size)
    

def main():
    seed_torch()
    args = parse_args()
    device = torch.device(args.device)
    
    config = load_yaml(f'{args.output_dir}/{args.name}/config.yml')
    if isinstance(config["input_list"], str):
        config["input_list"] = [int(i) for i in config["input_list"].split(",")]
    
    print('-'*20)
    for key in config.keys():
        print(f'{key}: {str(config[key])}')

    if "resnet" in args.name:
        size = (512, 512)
        model = get_resnet(size)
    else:
        print('-'*20)
        model = archs.__dict__[config['arch']](config['num_classes'], config['input_channels'], config['deep_supervision'], embed_dims=config['input_list'], no_kan=config['no_kan'])
        cudnn.benchmark = True
    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print("Total params: ", total_params)
    
    batch_size = args.batch_size if args.batch_size is not None else config['batch_size']
    print(f"Batch size: {batch_size}")
    dummy_input = torch.randn(batch_size, 3, config['input_h'], config['input_h']).to(device)
    with FlopTensorDispatchMode(model) as ftdm:
        # count forward flops
        res = model(dummy_input)
        flops_forward = copy.deepcopy(ftdm.flop_counts)
    gflops_forward = sum(flops_forward[""].values()) / 1e9
    print("GFLOPs: ", gflops_forward)
    # with FlopTensorDispatchMode(model) as ftdm:
    #     x = torch.rand(8, 40).cuda()
    #     res = model.dblock1[0].layer.fc1(x)
    #     flops_forward = copy.deepcopy(ftdm.flop_counts)
    # print("FLOPs: ", flops_forward)
    # macs, params = get_model_complexity_info(
    #       model,
    #       (3, 512, 512),
    #       as_strings=True,
    #       backend="pytorch",
    #       print_per_layer_stat=False,
    #       verbose=False,
    # )
    # print(f"MACs: {macs}")

    total_time = 0
    model = model.to(device)
    with torch.no_grad():
        for k in tqdm(range(ITERATIONS)):
            dummy_input = torch.randn(batch_size, 3, config['input_h'], config['input_h']).to(device)
            if k < WARM_UP:
                _ = model(dummy_input)
                continue
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            _ = model(dummy_input)
            end.record() # Waits for everything to finish running
            # Wait for GPU to finish
            torch.cuda.synchronize()
            # Calculate elapsed time
            elapsed_time = start.elapsed_time(end)
            total_time += elapsed_time
    average_time = (total_time / (ITERATIONS - WARM_UP)) / batch_size
    print(f"Average time: {average_time:.2f} ms")
    
    output_dir = args.output_dir
    
    if os.path.exists(os.path.join(output_dir, "complexity.csv")):
        overall_results = pd.read_csv(
            os.path.join(output_dir, "complexity.csv")
        )
    else:
        overall_results = pd.DataFrame(
            columns=["model", "GFLOPs", "params", "average_time"]
        )
        
    new_row = pd.DataFrame([{
        "model": args.name,
        "GFLOPs": gflops_forward,
        "params": total_params,
        "average_time": average_time,
    }])
    overall_results = pd.concat([overall_results, new_row], ignore_index=True)
    overall_results.to_csv(
        os.path.join(output_dir, "complexity.csv"), index=False
    )
    
if __name__ == '__main__':
    main()
