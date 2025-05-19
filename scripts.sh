python Seg_UKAN/train.py --arch UKAN --dataset roweeder --num_classes 3 --input_w 512 --input_h 512 --input_list 128,160,256 --name UKAN_128 --epochs 500 --data_dir data/weedmap/UKAN --loss FocalLoss -b 2 --early_stopping 20 --loss_weights
python Seg_UKAN/train.py --arch UKAN --dataset roweeder --num_classes 3 --input_w 512 --input_h 512 --input_list 64,80,128 --name UKAN_64 --epochs 500 --data_dir data/weedmap/UKAN --loss FocalLoss -b 2 --early_stopping 20 --loss_weights
python Seg_UKAN/train.py --arch UKAN --dataset roweeder --num_classes 3 --input_w 512 --input_h 512 --input_list 32,40,64 --name UKAN_32 --epochs 500 --data_dir data/weedmap/UKAN --loss FocalLoss -b 2 --early_stopping 20 --loss_weights
python main.py train --name segformer --epochs 500 --data_dir data/weedmap/UKAN --loss FocalLoss -b 2 --early_stopping 20 --loss_weights

python main.py val --name segformer --output_dir outputs
python Seg_UKAN/val-multiclass.py --name UKAN_32 --output_dir outputs
python Seg_UKAN/val-multiclass.py --name UKAN_64 --output_dir outputs
python Seg_UKAN/val-multiclass.py --name UKAN_128 --output_dir outputs

###

python Seg_UKAN/train.py --arch UKAN --dataset roweeder --num_classes 3 --input_w 512 --input_h 512 --input_list 128,160,256 --name UKAN_128_nw --epochs 500 --data_dir data/weedmap/UKAN --loss FocalLoss -b 2 --early_stopping 20
python Seg_UKAN/train.py --arch UKAN --dataset roweeder --num_classes 3 --input_w 512 --input_h 512 --input_list 64,80,128 --name UKAN_64_nw --epochs 500 --data_dir data/weedmap/UKAN --loss FocalLoss -b 2 --early_stopping 20
python Seg_UKAN/train.py --arch UKAN --dataset roweeder --num_classes 3 --input_w 512 --input_h 512 --input_list 32,40,64 --name UKAN_32_nw --epochs 500 --data_dir data/weedmap/UKAN --loss FocalLoss -b 2 --early_stopping 20
python main.py train --name segformer_nw --epochs 500 --data_dir data/weedmap/UKAN --loss FocalLoss -b 2 --early_stopping 20

python main.py val --name segformer_nw --output_dir outputs --device cuda:1
python Seg_UKAN/val-multiclass.py --name UKAN_32_nw --output_dir outputs --device cuda:1
python Seg_UKAN/val-multiclass.py --name UKAN_64_nw --output_dir outputs
python Seg_UKAN/val-multiclass.py --name UKAN_128_nw --output_dir outputs

python Seg_UKAN/train.py --arch UKAN --dataset roweeder --num_classes 3 --input_w 512 --input_h 512 --input_list 256,320,512 --name UKAN_256_nw_nokan --epochs 500 --data_dir data/weedmap/UKAN --loss FocalLoss -b 2 --early_stopping 20 --no_kan
python Seg_UKAN/train.py --arch UKAN --dataset roweeder --num_classes 3 --input_w 512 --input_h 512 --input_list 192,240,384 --name UKAN_192_nw_nokan --epochs 500 --data_dir data/weedmap/UKAN --loss FocalLoss -b 2 --early_stopping 20 --no_kan
python Seg_UKAN/train.py --arch UKAN --dataset roweeder --num_classes 3 --input_w 512 --input_h 512 --input_list 128,160,256 --name UKAN_128_nw_nokan --epochs 500 --data_dir data/weedmap/UKAN --loss FocalLoss -b 2 --early_stopping 20 --no_kan
python Seg_UKAN/train.py --arch UKAN --dataset roweeder --num_classes 3 --input_w 512 --input_h 512 --input_list 64,80,128 --name UKAN_64_nw_nokan --epochs 500 --data_dir data/weedmap/UKAN --loss FocalLoss -b 2 --early_stopping 20 --no_kan
python Seg_UKAN/train.py --arch UKAN --dataset roweeder --num_classes 3 --input_w 512 --input_h 512 --input_list 32,40,64 --name UKAN_32_nw_nokan --epochs 500 --data_dir data/weedmap/UKAN --loss FocalLoss -b 2 --early_stopping 20 --no_kan

python Seg_UKAN/val-multiclass.py --name UKAN_32_nw_nokan --output_dir outputs --device cuda:1
python Seg_UKAN/val-multiclass.py --name UKAN_64_nw_nokan --output_dir outputs --device cuda:1
python Seg_UKAN/val-multiclass.py --name UKAN_128_nw_nokan --output_dir outputs --device cuda:1
python Seg_UKAN/val-multiclass.py --name UKAN_192_nw_nokan --output_dir outputs --device cuda:1
python Seg_UKAN/val-multiclass.py --name UKAN_256_nw_nokan --output_dir outputs --device cuda:1