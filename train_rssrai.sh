CUDA_VISSIBLE_DEVICES=0,1 python train.py --backbone resnet_4c --lr 0.01 --workers 4 --epochs 40 --batch-size 10 --gpu-ids 0,1 --checkname deeplab-resnet_4c --eval-interval 1 --dataset rssrai