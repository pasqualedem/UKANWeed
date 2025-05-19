#! /data/cxli/miniconda3/envs/th200/bin/python
import argparse
import os
import random
import numpy as np

from ptflops import get_model_complexity_info
import torch
import torch.backends.cudnn as cudnn
from tqdm import tqdm
import yaml

import archs

from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

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


class EmbeddedSegformer(torch.nn.Module):
    def __init__(self, model: SegformerForSemanticSegmentation, processor: SegformerImageProcessor, input_size):
        super(EmbeddedSegformer, self).__init__()
        self.model = model
        self.processor = processor
        self.input_size = input_size
        
    def forward(self, x):
        # Forward pass through the model
        outputs = self.model(x)
        dimension_list = [self.input_size for _ in range(outputs.logits.shape[0])]
        outputs = self.processor.post_process_semantic_segmentation(outputs, target_sizes=dimension_list)
        outputs = torch.stack(outputs, dim=0)
        return outputs


def get_segformer(input_size):
    label2id = {'background':0, 'crop':1, 'weed':2}
    id2label = {0:'background', 1:'crop', 2:'weed'}
    # define model
    model = SegformerForSemanticSegmentation.from_pretrained("nvidia/mit-b0",
                                                            num_labels=3,
                                                            id2label=id2label,
                                                            label2id=label2id,
    )
    image_processor = SegformerImageProcessor(reduce_labels=False)  
    return EmbeddedSegformer(model, image_processor, input_size)  
    

def main():
    seed_torch()
    args = parse_args()
    device = torch.device(args.device)
    
    
    with open(f'{args.output_dir}/{args.name}/config.yml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    print('-'*20)
    for key in config.keys():
        print(f'{key}: {str(config[key])}')

    if "segformer" in args.name:
        size = (512, 512)
        model = get_segformer(size)
    else:
        print('-'*20)
        model = archs.__dict__[config['arch']](config['num_classes'], config['input_channels'], config['deep_supervision'], embed_dims=config['input_list'], no_kan=config['no_kan'])
        cudnn.benchmark = True
    print("Total params: ", sum(p.numel() for p in model.parameters()))
    macs, params = get_model_complexity_info(
          model,
          (3, 512, 512),
          as_strings=True,
          backend="pytorch",
          print_per_layer_stat=False,
          verbose=False,
    )
    print(f"MACs: {macs}")

    batch_size = args.batch_size if args.batch_size is not None else config['batch_size']
    print(f"Batch size: {batch_size}")
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
    
if __name__ == '__main__':
    main()
