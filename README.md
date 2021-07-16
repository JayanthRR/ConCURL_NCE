# Introduction 
Code for [Representation Learning for Clustering via Building Consensus](https://arxiv.org/pdf/2105.01289.pdf)

If you find this work useful to your, please cite :
```
@article{deshmukh2021representation,
  title={Representation Learning for Clustering via Building Consensus},
  author={Deshmukh, Aniket Anand and Regatti, Jayanth Reddy and Manavoglu, Eren and Dogan, Urun},
  journal={arXiv preprint arXiv:2105.01289},
  year={2021}
}
```

# Conda Environment

`conda env create -f concurl.yml`


# Reproduce Results
Download the models from [saved_models](https://drive.google.com/file/d/1iYw5mS8poqhaAOkeyGujQuuxEtUk8Xcc/view?usp=sharing) and store them in `saved_models` folder. The datasets need to be downloaded prior to executing the following command. 

`python extract_features_clustering_eval.py`


# Sample Commands

A sample command to run the algorithm is as follows. The commands for the best models are provided in `run.sh`.

```
CUDA_VISIBLE_DEVICES=0 python main.py \
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
--image-size 160 

```


# Results

| Dataset | Acc | NMI | ARI | 
| ---- | ---- | ---- | ---- | 
| ImageNet-10 | 0.958 | 0.907 | 0.909 |
| ImageNet-Dogs | 0.695 | 0.63 | 0.531 | 
| STL10 | 0.749 | 0.636 | 0.566 |
| CIFAR10 | 0.846 | 0.762 | 0.715 | 
| CIFAR100 | 0.479 | 0.468 | 0.303 |


# Acknowledgements
The code for the paper is built using the following two repositories: https://github.com/facebookresearch/swav, https://github.com/zhirongw/lemniscate.pytorch
