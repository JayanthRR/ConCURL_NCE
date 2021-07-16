#!/bin/bash


# CIFAR100

DATAPATH=<DATAPATH>
LOGPATH=<LOGPATH>

python main.py \
--use-no-grad \
--hidden-mlp 2048 \
--eval-freq 10 \
--trial 2 \
--workers 8 \
--use-consensus \
--use-rp \
--exp-name nce_consensus \
--alpha 0.0 \
--beta 1.0 \
--gamma 1.0 \
--use-torch-resnet \
--projection-dim 88 \
--n-transforms 16 \
--arch resnet18 \
--lr 0.06 \
--use-slightly-diff-views \
--optim SGD \
--nce-temp 0.35 \
--nce-k 4096 \
--n_clusters 20 \
--batch-size 128 \
--num-epochs 1 \
--datapath $DATAPATH \
--logdir $LOGPATH \
--image-size 32 


########################

# CIFAR10

DATAPATH=<DATAPATH>
LOGPATH=<LOGPATH>

python main.py \
--use-no-grad \
--hidden-mlp 2048 \
--eval-freq 10 \
--trial 2 \
--workers 8 \
--use-consensus \
--use-rp \
--exp-name nce_consensus \
--alpha 0.0 \
--beta 1.0 \
--gamma 1.0 \
--projection-dim 152 \
--n-transforms 64 \
--use-torch-resnet \
--arch resnet18 \
--use-slightly-diff-views \
--lr 0.015 \
--optim SGD \
--batch-size 128 \
--nce-temp 0.85 \
--nce-k 4096 \
--n_clusters 10 \
--num-epochs 1 \
--datapath $DATAPATH \
--logdir $LOGPATH \
--image-size 32 

##################################

# ImageNet-10

DATAPATH=<DATAPATH>
LOGPATH=<LOGPATH>

python main.py \
--use-no-grad \
--hidden-mlp 2048 \
--eval-freq 10 \
--trial 2 \
--workers 8 \
--use-consensus \
--use-rp \
--alpha 0.0 \
--beta 1.0 \
--gamma 1.0 \
--projection-dim 104 \
--n-transforms 1 \
--use-torch-resnet \
--arch resnet50 \
--use-slightly-diff-views \
--lr 0.03 \
--optim SGD \
--batch-size 128 \
--num-epochs 1 \
--nce-temp 1.0 \
--nce-k 4096 \
--n_clusters 10 \
--datapath $DATAPATH \
--logdir $LOGPATH \
--image-size 160


###############################

# ImageNet-Dogs

DATAPATH=<DATAPATH>
LOGPATH=<LOGPATH>

python main.py \
--use-no-grad \
--hidden-mlp 2048 \
--eval-freq 10 \
--trial 2 \
--workers 8 \
--use-consensus \
--use-rp \
--alpha 0.0 \
--beta 1.0 \
--gamma 1.0 \
--projection-dim 136 \
--n-transforms 64 \
--use-torch-resnet \
--arch resnet50 \
--use-slightly-diff-views \
--lr 0.06 \
--optim SGD \
--batch-size 128 \
--num-epochs 1 \
--nce-temp 0.5 \
--nce-k 4096 \
--n_clusters 15 \
--datapath $DATAPATH \
--logdir $LOGPATH \
--image-size 160 \

