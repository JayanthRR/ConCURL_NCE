import argparse
import os
import sys
import shutil
import time
import random
import warnings
import datetime
import sys
import json
import copy
import numpy as np
import pickle
from time import time
from tqdm import tqdm
import math
import csv

np.set_printoptions(threshold=sys.maxsize)

import sklearn
from sklearn import random_projection
from sklearn.cluster import KMeans, MiniBatchKMeans
from scipy.spatial import distance

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms

from utils import cluster_accuracy, main_cluster_metrics, write_csv, load_model
from losses import ByolLoss, SoftLoss, ConsensusLoss
from transformationGenerator import transformationGenerator
from dataset_utils import get_data_loaders

from model_utils import nce_resnet, concurl, torch_resnet, cifar_resnet

from lib.NCEAverage import NCEAverage
from lib.LinearAverage import LinearAverage
from lib.NCECriterion import NCECriterion
from lib.utils import AverageMeter

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_setting_for_different_clusterings = {}

log_folder = 'saved_models'

model_setting_for_different_clusterings['ImageNet-10'] = {}
model_setting_for_different_clusterings['ImageNet-10']['saved_experiment_folder'] = os.path.join(log_folder, 'ImageNet-10')
model_setting_for_different_clusterings['ImageNet-10']['dataset_path'] = '/ads-nfs/t-jregatti/data/ImageNet-10'

model_setting_for_different_clusterings['ImageNet-Dogs'] = {}
model_setting_for_different_clusterings['ImageNet-Dogs']['saved_experiment_folder'] = os.path.join(log_folder, 'ImageNet-Dogs')
model_setting_for_different_clusterings['ImageNet-Dogs']['dataset_path'] = '/ads-nfs/t-jregatti/data/ImageNet-Dogs'

model_setting_for_different_clusterings['CIFAR10'] = {}
model_setting_for_different_clusterings['CIFAR10']['saved_experiment_folder'] = os.path.join(log_folder, 'CIFAR10')
model_setting_for_different_clusterings['CIFAR10']['dataset_path'] =  '/ads-nfs/t-jregatti/data/CIFAR10'

model_setting_for_different_clusterings['CIFAR100'] = {}
model_setting_for_different_clusterings['CIFAR100']['saved_experiment_folder'] = os.path.join(log_folder, 'CIFAR100')
model_setting_for_different_clusterings['CIFAR100']['dataset_path'] = '/ads-nfs/t-jregatti/data/CIFAR100'



embedding_folder = os.path.join(log_folder, 'saved_embeddings')

if not os.path.exists(embedding_folder):
    os.makedirs(embedding_folder)

for datasetName in model_setting_for_different_clusterings.keys():
    print(datasetName)
    saved_experiment_folder = model_setting_for_different_clusterings[datasetName]['saved_experiment_folder']

    json_file_path = os.path.join(saved_experiment_folder, 'config.json')
    with open(json_file_path) as j_file:
        experiment_configs = json.load(j_file)

    args = AttrDict(experiment_configs)
    args.datapath = model_setting_for_different_clusterings[datasetName]['dataset_path']


    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%5")
    print('starting evaluation')

    saved_models = os.listdir(saved_experiment_folder)
    results_acc, results_nmi, results_ari = {}, {}, {}
    saved_model_epochs = {}
    for saved_model_name in tqdm(saved_models):
        # print('model_name', saved_model_name)
        if 'linear' in saved_model_name:
            continue
        elif 'nce_embeddings' in saved_model_name:
            # Evaluate only nce_embeddings.pth.tar models, and ignore others.

            saved_model_path = os.path.join(saved_experiment_folder, saved_model_name)

            results_acc[saved_model_name], results_nmi[saved_model_name], results_ari[saved_model_name], saved_model_epochs[saved_model_name], nce_embeddings, labels = main_cluster_metrics(args, saved_model_path, 1, return_embeddings=True)
            print('results_acc: ', results_acc[saved_model_name])
            print('results_nmi: ', results_nmi[saved_model_name])
            print('results_ari: ', results_ari[saved_model_name])

            # np.save(os.path.join(embedding_folder, datasetName+'_rep.npy'), nce_embeddings)
            # np.save(os.path.join(embedding_folder, datasetName+'_labels.npy'), labels)

    save_folder = os.path.join(*[saved_experiment_folder, 'evaluation_results'])
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    with open(os.path.join(save_folder, 'saved_model_epochs.json'),'w') as f:
        json.dump(saved_model_epochs, f, indent=4)

    with open(os.path.join(save_folder, 'accuracy.json'),'w') as f:
        json.dump(results_acc, f, indent=4)

    write_csv(results_acc, file_name=os.path.join(save_folder, 'accuracy.xlsx'))

    with open(os.path.join(save_folder, 'nmi.json'),'w') as f:
        json.dump(results_nmi, f, indent=4)
    write_csv(results_nmi, file_name=os.path.join(save_folder, 'nmi.xlsx'))

    with open(os.path.join(save_folder, 'ari.json'),'w') as f:
        json.dump(results_ari, f, indent=4)
    write_csv(results_ari, file_name=os.path.join(save_folder, 'ari.xlsx'))

print('evaluation complete')


