import argparse
from SegFormer.train import main as train_main
from SegFormer.val import main as val_main
from Seg_UKAN.utils import str2bool
from Seg_UKAN import losses

LOSS_NAMES = losses.__all__

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("command", choices=["train", "val"], help="Command to execute: 'train' or 'val'")
    parser.add_argument('--name', default=None,
                        help='model name: (default: arch+timestamp)')
    parser.add_argument('--epochs', default=400, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch_size', default=8, type=int,
                        metavar='N', help='mini-batch size (default: 16)')

    parser.add_argument('--dataseed', default=2981, type=int,
                        help='')
    
    parser.add_argument('--deep_supervision', default=False, type=str2bool)
    parser.add_argument('--input_channels', default=3, type=int,
                        help='input channels')
    # loss
    parser.add_argument('--loss', default='BCEDiceLoss',
                        choices=LOSS_NAMES,
                        help='loss: ' +
                        ' | '.join(LOSS_NAMES) +
                        ' (default: BCEDiceLoss)')
    parser.add_argument('--loss_weights', default=False, action='store_true',
                        help='apply loss weight')
    
    # dataset  
    parser.add_argument('--data_dir', default='inputs', help='dataset dir')

    parser.add_argument('--output_dir', default='outputs', help='ouput dir')


    # optimizer
    parser.add_argument('--optimizer', default='Adam',
                        choices=['Adam', 'SGD'],
                        help='loss: ' +
                        ' | '.join(['Adam', 'SGD']) +
                        ' (default: Adam)')

    parser.add_argument('--lr', '--learning_rate', default=0.00006, type=float,
                        metavar='LR', help='initial learning rate')

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
    parser.add_argument('--device', default='cuda:0', type=str,
                        help='device to use for training / testing')
    
    return parser.parse_args()

def main():
    args = parse_args()

    if args.command == "train":
        train_main(args)
    elif args.command == "val":
        val_main(args)

if __name__ == "__main__":
    main()