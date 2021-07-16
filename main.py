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


parser = argparse.ArgumentParser(description='BYOL')

parser.add_argument('--git-log', default=' ')
parser.add_argument('--datapath', default='/home/cc/data/ImageNet-10/')
parser.add_argument('--logdir', default='/home/cc/NCE_test/ImageNet-10/')
parser.add_argument('--batch-size', default=128, type=int)
parser.add_argument('--trial', default=0, type=int)
parser.add_argument('--num-epochs', default=5, type=int)
parser.add_argument('--arch', default='resnet18', choices=['resnet18','resnet34', 'resnet50', 'resnet101'])
parser.add_argument('--use-torch-resnet', default=False, action='store_true')
parser.add_argument('--use-train-test', default=False, action='store_true')

parser.add_argument('--exp-name', default='baseline')
parser.add_argument('--config-num', default=0, type=int)
parser.add_argument('--n_clusters', default=10, type=int)

parser.add_argument('--optim', default='SGD')
parser.add_argument('--lr', default=0.5,type=float)
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float, metavar='W', help='weight decay (default: 5e-4)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')

# alpha -> Soft, beta-> BYOL, gamma -> Consensus
# TODO: Paper has a slightly different combination of alpha, beta, gamma

parser.add_argument('--alpha', default=0.0, type=float)
parser.add_argument('--beta', default=1.0, type=float)
parser.add_argument('--gamma', default=0.0, type=float)

parser.add_argument('--use-consensus', action='store_true', default=False)
parser.add_argument('--n-transforms', default=10, type=int)
parser.add_argument('--use-rp', action='store_true', default=False)
parser.add_argument('--projection-dim', default=256, type=int)
parser.add_argument('--use-sobel', action='store_true', default=False)
parser.add_argument('--include-rgb', action='store_true', default=False)

parser.add_argument('--reinit-transforms', default=False, action='store_true')
parser.add_argument('--n-overclusters', default=40, type=int)
parser.add_argument('--use-overclustering', default=False, action='store_true')

parser.add_argument('--workers', default=8, type=int)
parser.add_argument('--checkpt-freq', default=50, type=int)
parser.add_argument('--eval-freq', default=10, type=int)

parser.add_argument('--swav-temp', default=0.1, type=float)

parser.add_argument('--use-no-grad', action='store_true', default=False)
parser.add_argument('--no-shared-params', action='store_true', default=False)
parser.add_argument('--hidden-mlp', default=1024, type=int)
parser.add_argument('--out-dim', default=256, type=int)
parser.add_argument('--image-size', default=224, type=int)

parser.add_argument('--seed', default=1111, type=int)

parser.add_argument('--perform-evaluation', default=False, action='store_true')

parser.add_argument('--name-args', 
                    default=[
                        'arch', 'alpha', 'beta', 'gamma', 'use_sobel', 'include_rgb',
                        'n_transforms', 'use_rp', 'projection_dim',
                        'n_clusters', 'lr', 'batch_size', 'optim',
                        'image_size', 'nce_temp', 'use_torch_resnet'
                            ],
                    nargs="+")

parser.add_argument('--restore-from-ckpt', default=False, action='store_true')
parser.add_argument('--restore-path', default='/home/')
parser.add_argument('--time-stamp', default=' ')

parser.add_argument('--evaluate_knn', default=False, action='store_true')
parser.add_argument('--low-dim', default=128, type=int,
                    metavar='D', help='feature dimension')
parser.add_argument('--nce-k', default=4096, type=int,
                    metavar='K', help='number of negative samples for NCE')
parser.add_argument('--nce-temp', default=0.3, type=float)
parser.add_argument('--nce-m', default=0.5, type=float,
                    help='momentum for non-parametric updates')
parser.add_argument('--use-slightly-diff-views', default=False, action='store_true')


args = parser.parse_args()

with open('seeds.pkl', 'rb') as seeds_file:
    saved_seeds = pickle.load(seeds_file)

def fix_random_seeds(trial=0):
    """
    Fix random seeds.
    """
    seed = saved_seeds['main_seeds'][trial]
    args.seed = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def save_checkpoint(state, folder, filename='checkpoint.pth.tar'):
    if not os.path.exists(folder):
        os.makedirs(folder)
    try:
        torch.save(state, os.path.join(folder, filename))
    except:
        pass
    
def adjust_learning_rate(optimizer, epoch, decay=0.1):

    if epoch in [600, 950, 1300, 1650, 2000]:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * decay        
    else:
        pass

    
def main(args, timestamp):

    fix_random_seeds(args.trial)

    # create model
    model = load_model(args)

    n_clusters = args.n_clusters
    swav_temp = args.swav_temp
    nce_temp = args.nce_temp
    epsilon = 0.05

    if (args.alpha, args.beta, args.gamma) == (0, 1, 0):
        nce_baseline = True
    else:
        nce_baseline = False

    # Data loading code
    train_loader, train_loader_for_eval, test_loader = get_data_loaders(
        args.datapath, image_size=args.image_size, batch_size=args.batch_size, 
        get_train=True, nce_baseline=nce_baseline, use_train_test=args.use_train_test,
        use_slightly_diff_views=args.use_slightly_diff_views
    )

    # define lemniscate and loss function (criterion)
    ndata = train_loader.dataset.__len__()

    if args.nce_k > 0:
        lemniscate = NCEAverage(args.low_dim, ndata, args.nce_k, args.nce_temp, args.nce_m).cuda()
        criterion = NCECriterion(ndata).cuda()
    else:
        lemniscate = LinearAverage(args.low_dim, ndata, args.nce_temp, args.nce_m).cuda()
        criterion = nn.CrossEntropyLoss().cuda()

        
    if args.optim == 'SGD':
        opt = torch.optim.SGD(
            model.parameters(), lr=args.lr, momentum=args.momentum, 
            weight_decay=args.weight_decay
        )
    else:
        opt = torch.optim.Adam(
            model.parameters(), lr=args.lr,
            weight_decay=args.weight_decay
        )

    # Restore from checkpoint
    if args.restore_from_ckpt:
        loaded_model = torch.load(args.model_path)
        model_state_dict = loaded_model['state_dict']
        opt_state_dict = loaded_model['optimizer']
        
        for k, v in model.state_dict().items():

            if k not in list(model_state_dict):
                print("not correct model")

            elif model_state_dict[k].shape != v.shape:
                print('key "{}" is of different shape in model and provided state dict'.format(k))
                model_state_dict[k] = v

        try:
            model.load_state_dict(model_state_dict, strict=True)
            opt.load_state_dict(opt_state_dict)
        except:
            print("state dict not correctly loaded")
            raise

        start_epoch = loaded_model['epoch']
        cum_itr = start_epoch * len(train_loader)

    else:
        start_epoch = 0
        cum_itr = 0

    cum_metrics = {}
    ############################################################################################################################
    experiment_path = os.path.join(args.logdir, args.exp_name, 'config_%d'%args.config_num, 'trial_%d'%args.trial, timestamp)
    if not os.path.exists(experiment_path):
        os.makedirs(experiment_path)

    writer_path = os.path.join(experiment_path, 'runs')
    writer = SummaryWriter(writer_path)
    
    with open(os.path.join(experiment_path,'config.json'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    tqdm_file = os.path.join(experiment_path, 'tqdm_file.txt')
    stdout_file = os.path.join(experiment_path, 'stdout_file.txt')
    orig_stdout = sys.stdout
    f = open(stdout_file, 'a+')
    sys.stdout = f
    tqdmf = open(tqdm_file, "a+")
    ############################################################################################################################

    best_prec1 = 0
    
    for epoch in range(start_epoch, args.num_epochs):

        if args.optim=='SGD':
            adjust_learning_rate(opt, epoch)

        # train for one epoch
        cum_itr = train(train_loader, model, lemniscate, criterion, opt, epoch, cum_itr, tqdmf, writer, swav_temp, nce_baseline=nce_baseline)

        #### EVALUATION ######
        ######################################################################################################################

        if (epoch % args.eval_freq == 0) or (epoch == args.num_epochs - 1):
            # compute cluster accuracy as a dictionary

            with torch.no_grad():
                t1 = time()
                clust_acc, clust_nmi, clust_ari = cluster_accuracy(train_loader_for_eval, model, n_clusters=n_clusters, use_kmeans=True)
                t2 = time()

                # print("epoch: %d, Time taken: %f"%(epoch+1, t2-t1))
                # print('cluster accuracy on train data')
                # print(clust_acc, clust_nmi, clust_ari)

                for key, val in clust_acc.items():
                    
                    writer.add_scalar('ClusterAcc/'+key, val['mean'], epoch+1)
                    max_key = cum_metrics['ClusterAcc/'+key] if 'ClusterAcc/'+key in cum_metrics else 0
                    cum_metrics['ClusterAcc/'+key] = max(val['mean'], max_key)
                    writer.add_scalar('Best/ClusterAcc_'+key, cum_metrics['ClusterAcc/'+key], epoch+1)

                for key, val in clust_nmi.items():
                    writer.add_scalar('ClusterNMI/'+key, val['mean'], epoch+1)
                    max_key = cum_metrics['ClusterNMI/'+key] if 'ClusterNMI/'+key in cum_metrics else 0
                    cum_metrics['ClusterNMI/'+key] = max(val['mean'], max_key)
                    writer.add_scalar('Best/ClusterNMI_'+key, cum_metrics['ClusterNMI/'+key], epoch+1)

                for key, val in clust_ari.items():
                    writer.add_scalar('ClusterARI/'+key, val['mean'], epoch+1)
                    max_key = cum_metrics['ClusterARI/'+key] if 'ClusterARI/'+key in cum_metrics else 0
                    cum_metrics['ClusterARI/'+key] = max(val['mean'], max_key)
                    writer.add_scalar('Best/ClusterARI_'+key, cum_metrics['ClusterARI/'+key], epoch+1)

                for cum_key, _ in clust_acc.items():

                    folder_name = os.path.join(experiment_path, 'checkpoints')

                    if (cum_metrics['ClusterAcc/'+cum_key] == clust_acc[cum_key]['mean']):
                        save_checkpoint(
                            {
                                'epoch': epoch + 1,
                                'state_dict': model.state_dict(),
                                'optimizer' : opt.state_dict(),
                                'rand_transforms': model.rand_transforms,
                                'clust_acc' : clust_acc,
                                'clust_nmi' : clust_nmi,
                                'clust_ari' : clust_ari,
                                'cum_itr': cum_itr,
                            },
                            folder=folder_name,
                            filename='best_%s.pth.tar'%(cum_key)
                        )

        folder_name = os.path.join(experiment_path, 'checkpoints')

        save_checkpoint(
            {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer' : opt.state_dict(),
                'rand_transforms':model.rand_transforms,
                'cum_itr': cum_itr,
            },
            folder=folder_name,
            filename='latest.pth.tar'
        )
    
    ############################################################################
    # Use best models so far and compute cluster accuracy on them on both training and test loaders

    if args.perform_evaluation:
        print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%5")
        print('starting evaluation')
        
        folder_name = os.path.join(*[experiment_path, 'checkpoints'])
        saved_models = os.listdir(folder_name)
        results_acc, results_nmi, results_ari = {}, {}, {}
        saved_model_epochs = {}
        for saved_model_name in tqdm(saved_models):
            print('model_name', saved_model_name)
            if 'linear' in saved_model_name:
                continue
            else:
                saved_model_path = os.path.join(folder_name, saved_model_name)
                # evaluating for only 1 trial as clustering might take a lot of time

                results_acc[saved_model_name], results_nmi[saved_model_name], results_ari[saved_model_name], saved_model_epochs[saved_model_name] = main_cluster_metrics(args, saved_model_path, 1)
                print('results_acc: ', results_acc[saved_model_name])
                print('results_nmi: ', results_nmi[saved_model_name])
                print('results_ari: ', results_ari[saved_model_name])
        
        save_folder = os.path.join(*[experiment_path, 'evaluation_results'])
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

    
    writer.close()
    sys.stdout = orig_stdout
    f.close()

    

def train(train_loader, model, lemniscate, criterion, opt, epoch, cum_itr, tqdmf, writer, swav_temp, nce_baseline=False):
    
    data_iterator = tqdm(
        train_loader,
        leave=True,
        unit="batch",
        file=tqdmf,
        postfix={
            "epo": epoch,
            "avglss": "%.6f" %0.0,
            "lss": "%.6f" % 0.0,
            "nce":"%.6f" % 0.0,
            "sf":"%.6f"%0.0,
            "cons":"%.6f" % 0.0,            
        },
        disable=False,
    )
    avg_loss = 0
        
    model.train()

    if args.use_consensus and args.reinit_transforms:
        model.update_transforms()

    for ind, (batch, _, index) in enumerate(data_iterator):

        cum_itr += 1

        # normalize centroids/prototypes before forward pass
        with torch.no_grad():
            w = model.prototypes.weight.data.clone()
            w = nn.functional.normalize(w, dim=1, p=2)
            model.prototypes.weight.copy_(w)

        batch_1, batch_2 = batch

        batch_1 = batch_1.cuda(non_blocking=True)
        # NOTE: Currently only batch_1 is used to compute NCE loss
        features_one, outcodes_one, rand_outs_one = model.forward(batch_1)

        batch_2 = batch_2.cuda(non_blocking=True)
        features_two, outcodes_two, rand_outs_two = model.forward(batch_2)

        #########################################################################################################################
        ## NCE loss
        index = index.cuda()

        # compute output
        nce_output = lemniscate(features_one, index)
        nce_loss = criterion(nce_output, index)

        #########################################################################################################################
        ## Soft loss

        softloss, q_one, q_two = SoftLoss(outcodes_one, outcodes_two, alpha=args.alpha, temperature=swav_temp)

        #########################################################################################################################
        ## Consensus loss

        if (args.use_consensus) and (args.gamma > 0):
            consensus_loss = ConsensusLoss(args.gamma, outcodes_one, outcodes_two, rand_outs_one, rand_outs_two, q_one, q_two, temperature=swav_temp)
        else:
            consensus_loss = torch.tensor(0)

        #########################################################################################################################
        ## Total loss
        loss = args.beta *nce_loss + args.alpha * softloss + args.gamma * consensus_loss
        
        opt.zero_grad()
        loss.backward()

        opt.step()
        with torch.no_grad():
            avg_loss += loss.item()

        data_iterator.set_postfix(
            epo=epoch,
            lss="%.6f" % float(loss.item()),
            avglss="%.6f" % float(avg_loss/(ind+1)),
            bk="%.6f" % nce_loss.item(),
            sf="%.6f" % softloss.item(),
            cons="%.6f" % consensus_loss.item(),
        )
        writer.add_scalar('Loss/iter', float(loss.item()), cum_itr)
        writer.add_scalar('Loss/avg', float(avg_loss/(ind+1)), cum_itr)
        writer.add_scalar('Loss/nce', nce_loss.item(), cum_itr)
        writer.add_scalar('Loss/sf', softloss.item(), cum_itr)
        writer.add_scalar('Loss/cons', consensus_loss.item(), cum_itr)

        for param_grp in opt.param_groups:
            lr0 = param_grp["lr"]
        writer.add_scalar('Loss/learningrate', lr0, cum_itr)

    return cum_itr
        
        


if __name__=="__main__":
    
    time_stamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    
    if args.restore_from_ckpt:
        num_epochs = args.num_epochs
        restore_path = args.restore_path
        time_stamp = args.time_stamp
        restore_path_files = os.listdir(args.restore_path)
        
        if 'config.json' not in restore_path_files:
            raise FileNotFoundError
        else:
            with open(args.restore_path + '/config.json', 'r') as f:
                new_args = json.load(f)
            
            for key, val in new_args.items():
                args.__dict__[key] = val
            args.num_epochs = num_epochs
            args.restore_from_ckpt = True
            args.restore_path = restore_path
            args.time_stamp = time_stamp
            
        if 'latest.pth.tar' not in os.listdir(args.restore_path + '/checkpoints/'):
            raise FileNotFoundError
        else:
            args.model_path = args.restore_path + '/checkpoints/latest.pth.tar'
    else:
        args.time_stamp = time_stamp
    
    key_args = args.name_args
        
    time_stamp += "".join(["{%s}_" for _ in range(len(key_args))])
    
    tup=[]
    for key_arg in key_args:
        tup.extend([args.__dict__[key_arg]])

    time_stamp = time_stamp % tuple(tup)
    # time_stamp = time_stamp % tuple(args.__dict__[key_arg] for key_arg in key_args)
    main(args, time_stamp)
