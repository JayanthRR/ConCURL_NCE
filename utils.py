import sklearn
from sklearn import random_projection
from sklearn.cluster import KMeans, MiniBatchKMeans
from scipy.spatial import distance
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from sklearn.metrics.cluster import adjusted_mutual_info_score
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics.cluster import adjusted_rand_score
from collections import Counter
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
import os

from transformationGenerator import transformationGenerator
from dataset_utils import get_data_loaders
from model_utils import concurl, nce_resnet, torch_resnet, cifar_resnet

import pandas as pd

from tqdm import tqdm
import time
import random
import copy


def distributed_sinkhorn(Q, nmb_iters):
    with torch.no_grad():
        sum_Q = torch.sum(Q)
        # dist.all_reduce(sum_Q)
        Q /= sum_Q

        u = torch.zeros(Q.shape[0]).cuda(non_blocking=True)
        r = torch.ones(Q.shape[0]).cuda(non_blocking=True) / Q.shape[0]
        c = torch.ones(Q.shape[1]).cuda(non_blocking=True) / ( Q.shape[1])

        curr_sum = torch.sum(Q, dim=1)
        #  dist.all_reduce(curr_sum)

        for it in range(nmb_iters):
            u = curr_sum
            Q *= (r / u).unsqueeze(1)
            Q *= (c / torch.sum(Q, dim=0)).unsqueeze(0)
            curr_sum = torch.sum(Q, dim=1)
            # dist.all_reduce(curr_sum)
        return (Q / torch.sum(Q, dim=0, keepdim=True)).t().float()

def load_model(args):

    if 'CIFAR' in args.datapath:
        print("=> creating torch resnet model '{}'".format(args.arch))
        if args.arch == 'resnet18':
            model = cifar_resnet.ResNet18(low_dim=args.low_dim)
        elif args.arch == 'resnet34':
            model = cifar_resnet.ResNet34(low_dim=args.low_dim)
        elif args.arch == 'resnet50':
            model = cifar_resnet.ResNet50(low_dim=args.low_dim)
        else:
            raise NotImplementedError
    else:
        if args.use_torch_resnet:        
            print("=> creating torch resnet model '{}'".format(args.arch))
            if args.arch == 'resnet18':
                model = torch_resnet.resnet18(num_classes=args.low_dim)
            elif args.arch == 'resnet34':
                model = torch_resnet.resnet34(num_classes=args.low_dim)
            elif args.arch == 'resnet50':
                model = torch_resnet.resnet50(num_classes=args.low_dim)
            else:
                raise NotImplementedError
        else:
            print("=> creating nce resnet model '{}'".format(args.arch))
            if args.arch == 'resnet18':
                model = nce_resnet.resnet18(low_dim=args.low_dim)
            elif args.arch == 'resnet34':
                model = nce_resnet.resnet34(low_dim=args.low_dim)
            elif args.arch == 'resnet50':
                model = nce_resnet.resnet50(low_dim=args.low_dim)
            else:
                raise NotImplementedError
    
    if (args.alpha, args.beta, args.gamma) == (0, 1, 0):

        nce_baseline = True
    else:
        nce_baseline = False

    if nce_baseline or (args.alpha, args.beta, args.gamma) == (1, 0, 0):

        # NCE_baseline/ Soft Baseline uses only 3 input channels
        # model.conv1 = nn.Conv2d(2, 64, kernel_size=(7,7), stride=(2,2), padding=(3,3), bias=False)
        pass

    else:
        if 'CIFAR' in args.datapath:

            if args.use_sobel:
                if args.include_rgb:
                    # When using Sobel and RGB together, the in channels needs to be 5 (and cifar network has a different first conv layer)
                    model.conv1 = nn.Conv2d(5, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                else:
                    model.conv1 = nn.Conv2d(2, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

        else:

            if args.use_sobel:
                if args.include_rgb:
                    model.conv1 = nn.Conv2d(5, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
                else:
                    model.conv1 = nn.Conv2d(2, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    

    if args.use_consensus:
        transform_generator = transformationGenerator(args.n_transforms, args.out_dim, args.projection_dim, args.num_epochs, args.use_rp)
        rand_transforms = transform_generator.getTransformations()
    else:
        transform_generator, rand_transforms = None, None

        
    model = concurl.Unsup(
        model, args.n_clusters, out_dim=args.out_dim,
        normalize=True,
        arch=args.arch,
        hidden_mlp=2048, rand_transforms=rand_transforms,
        use_no_grad=False,
        use_sobel=args.use_sobel,
        include_rgb=args.include_rgb,
        transform_generator=transform_generator
    )

    # make sure resnet model is loaded correctly based on args
    # print(model)
    # print(args.use_torch_resnet)

    model = model.cuda()

    return model


def load_model_from_checkpoint(model, model_path, return_epoch=False):

    #####################################################

    loaded_model = torch.load(model_path)
    state_dict = loaded_model['state_dict']
    ep = loaded_model['epoch']
    
    for k, v in model.state_dict().items():

        if k not in list(state_dict):
            print("not correct model")

        elif state_dict[k].shape != v.shape:
            print('key "{}" is of different shape in model and provided state dict'.format(k))
            state_dict[k] = v

    # load state dict into the model
    model.load_state_dict(state_dict, strict=False)

    if 'rand_transforms' in loaded_model.keys():
        model.rand_transforms = loaded_model['rand_transforms']

    if return_epoch:
        return model, ep

    return model


def cluster_accuracy_helper(y_true, y_predicted, cluster_number=None):
    """
    Calculate clustering accuracy after using the linear_sum_assignment function in SciPy to
    determine reassignments.

    Args:
        y_true ([type]):  list of true cluster numbers, an integer array 0-indexed
        y_predicted ([type]): list  of predicted cluster numbers, an integer array 0-indexed
        cluster_number ([type], optional):number of clusters, if None then calculated from input

    Returns:
        [list of dicts]: accuracy, ami score, ari score, reassignment, prediction_vector
    """
        
    if cluster_number is None:
        cluster_number = (
            max(y_predicted.max(), y_true.max()) + 1
        )  # assume labels are 0-indexed
    count_matrix = np.zeros((cluster_number, cluster_number), dtype=np.int64)

    for i in range(y_true.size):
        count_matrix[y_predicted[i], y_true[i]] += 1

    row_ind, col_ind = linear_sum_assignment(count_matrix.max() - count_matrix)
    
    # reassignment maps 
    reassignment = dict(zip(row_ind, col_ind))
    reverse_reassignment = dict(zip(col_ind, row_ind))

    #refer to scipy docs of linear_sum_assignment for example on how this is used
    accuracy = count_matrix[row_ind, col_ind].sum() / y_predicted.size

    prediction_vec = [reassignment[y_predicted[ind]] for ind in range(len(y_predicted))]

    ## new function based on majority vote: assign majority vote class to all data samples of this cluster
    # form cluster indices
    #TODO:
    cluster_index_to_idx = {}
    for i in range(y_predicted.size):
        if y_predicted[i] in cluster_index_to_idx:
            cluster_index_to_idx[y_predicted[i]].append(i)
        else:
            cluster_index_to_idx[y_predicted[i]] = []
            cluster_index_to_idx[y_predicted[i]].append(i)
    
    incorrect = 0
    for cind in range(cluster_number):
        if cind in cluster_index_to_idx:
            
            ctr = Counter(y_true[cluster_index_to_idx[cind]])
            incorrect += len(cluster_index_to_idx[cind]) - ctr.most_common()[0][1]
            
    majority_accuracy=1 - (incorrect/y_predicted.size)
    
    ami_score = adjusted_mutual_info_score(y_true, prediction_vec)
    ari_score = adjusted_rand_score(y_true, prediction_vec)
    nmi_score = normalized_mutual_info_score(y_true, prediction_vec)
    
    return accuracy, majority_accuracy, nmi_score, ami_score, ari_score, torch.tensor(prediction_vec).float()


def cluster_accuracy(dataloader, model, return_actual=True, n_clusters=10, use_kmeans=False):

    model.eval()
    nce_embeddings, clust_head, outcode, q_code, actual = get_embeddings(model, dataloader)
    
    accuracy, nmi, ari = evaluation_helper(1, n_clusters, actual, nce_embeddings, clust_head, outcode, q_code, use_kmeans=True)
    
    return accuracy, nmi, ari


def get_embeddings(model, dataloader):
    
    model.eval()
    nce_embeddings = []
    clust_head, outcode = [], []

    actual = []
    with torch.no_grad():
        print('computing features')
        # FIXME: Make sure the data loader is chosen for evaluation
        for i, (images, target, _) in tqdm(enumerate(dataloader)):

            actual.append(target)

            images = images.cuda(non_blocking=True)
            
            features, outcodes, _ = model.forward(images)            

            # features has rep and proj layer values
            
            nce_embeddings.append(features.detach().cpu())
                    
            clust_head.append(outcodes['clust_head'].detach().cpu())
            outcode.append(outcodes['cTz'].detach().cpu())

            
        actual_for_hist = torch.cat(actual).float()
        actual = torch.cat(actual).long().cpu().numpy()

        nce_embeddings = torch.cat(nce_embeddings).numpy()        

        clust_head = torch.cat(clust_head).numpy()
        outcode = torch.cat(outcode)

        epsilon = 0.05
        q_code = torch.exp(outcode.cuda() / epsilon).t()
        q_code = distributed_sinkhorn(q_code, 3).cpu()
    
    return nce_embeddings, clust_head, outcode, q_code, actual


def evaluation_helper(num_trials, n_clusters, actual, nce_embeddings, clust_head, outcode, q_code, use_kmeans=False):

    base_dict = {'list':[], 'mean':0,'std':0}

    accuracy_dict = {
        'clust_head': copy.deepcopy(base_dict),
        'outcode': copy.deepcopy(base_dict),
        'q_code': copy.deepcopy(base_dict),
        'nce_embeddings': copy.deepcopy(base_dict),
    }

    nmi_dict = {
        'clust_head': copy.deepcopy(base_dict),
        'outcode': copy.deepcopy(base_dict),
        'q_code': copy.deepcopy(base_dict),
        'nce_embeddings': copy.deepcopy(base_dict),
    }

    ari_dict = {
        'clust_head': copy.deepcopy(base_dict),
        'outcode': copy.deepcopy(base_dict),
        'q_code': copy.deepcopy(base_dict),
        'nce_embeddings': copy.deepcopy(base_dict),
    }
    
    for trial in range(num_trials):
        if use_kmeans:

            # kmeans on clust_head embeddings
            if clust_head.shape[0] > 7000:
                clust_head_kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=trial, n_init=20, batch_size=6000)
            else:
                clust_head_kmeans = KMeans(n_clusters=n_clusters, random_state=trial, n_init=20)
            
            predicted_clust_head = clust_head_kmeans.fit_predict(clust_head)
            clust_head_accuracy, clust_head_maj_acc, clust_head_nmi, clust_head_ami, clust_head_ari, _ = cluster_accuracy_helper(actual, predicted_clust_head)
            
            # kmeans on target_proj embeddings
            if nce_embeddings.shape[0] > 7000:
                nce_embeddings_kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=trial, n_init=20, batch_size=6000)
            else:
                nce_embeddings_kmeans = KMeans(n_clusters=n_clusters, random_state=trial, n_init=20)

            predicted_nce_embeddings = nce_embeddings_kmeans.fit_predict(nce_embeddings)
            nce_embeddings_accuracy, nce_embeddings_maj_acc, nce_embeddings_nmi, nce_embeddings_ami, nce_embeddings_ari, _ = cluster_accuracy_helper(actual, predicted_nce_embeddings)


        else:
            clust_head_accuracy, clust_head_maj_acc, clust_head_nmi, clust_head_ami, clust_head_ari = 0, 0, 0, 0, 0
            nce_embeddings_accuracy, nce_embeddings_maj_acc, nce_embeddings_nmi, nce_embeddings_ami, nce_embeddings_ari = 0, 0, 0, 0, 0

        # argmax on outcode
        predicted_outcode = outcode.argmax(dim=1).numpy()

        # argmax on q_code
        predicted_q_code = q_code.argmax(dim=1).numpy()
        
        ####################
        
        outcode_accuracy, outcode_maj_acc, outcode_nmi, outcode_ami, outcode_ari, _ = cluster_accuracy_helper(actual, predicted_outcode)
        q_code_accuracy, q_code_maj_acc, q_code_nmi, q_code_ami, q_code_ari, _ = cluster_accuracy_helper(actual, predicted_q_code)
        
        accuracy_dict['outcode']['list'].append(outcode_accuracy)
        accuracy_dict['q_code']['list'].append(q_code_accuracy)
        accuracy_dict['clust_head']['list'].append(clust_head_accuracy)
        accuracy_dict['nce_embeddings']['list'].append(nce_embeddings_accuracy)

        nmi_dict['outcode']['list'].append(outcode_nmi)
        nmi_dict['q_code']['list'].append(q_code_nmi)
        nmi_dict['clust_head']['list'].append(clust_head_nmi)
        nmi_dict['nce_embeddings']['list'].append(nce_embeddings_nmi)

        ari_dict['outcode']['list'].append(outcode_ari)
        ari_dict['q_code']['list'].append(q_code_ari)
        ari_dict['clust_head']['list'].append(clust_head_ari)
        ari_dict['nce_embeddings']['list'].append(nce_embeddings_ari)
        

    for key in accuracy_dict.keys():
        accuracy_dict[key]['mean'], accuracy_dict[key]['std'] = np.mean(accuracy_dict[key]['list']), np.std(accuracy_dict[key]['list'])
        nmi_dict[key]['mean'], nmi_dict[key]['std'] = np.mean(nmi_dict[key]['list']), np.std(nmi_dict[key]['list'])
        ari_dict[key]['mean'], ari_dict[key]['std'] = np.mean(ari_dict[key]['list']), np.std(ari_dict[key]['list']) 
        
    return accuracy_dict, nmi_dict, ari_dict
        

def main_cluster_metrics(args, model_path, num_trials, return_embeddings=False):
    
    # return embeddings only for nce_embeddings
    # instantiate model according to args and load model from disk
    # Todo: include adding rand_transforms inside load_model_
    
    if (args.alpha, args.beta, args.gamma) == (0, 1, 0):

        nce_baseline = True
    else:
        nce_baseline = False

    learner = load_model(args)
    learner, ep = load_model_from_checkpoint(learner, model_path, return_epoch=True)

    train_loader, test_loader = get_data_loaders(
        args.datapath, image_size=args.image_size, batch_size=args.batch_size, 
        get_train=False, nce_baseline=nce_baseline, use_train_test=args.use_train_test,
        use_slightly_diff_views=args.use_slightly_diff_views
    )

    # if nce_baseline:
    #     train_loader, test_loader = get_data_loaders(
    #         args.datapath, image_size=args.image_size, batch_size=args.batch_size, 
    #         get_train=False, nce_baseline=True, use_train_test=args.use_train_test
    #     )
    # else:
    #     train_loader, test_loader = get_data_loaders(
    #         args.datapath, image_size=args.image_size, batch_size=args.batch_size, 
    #         get_train=False, use_train_test=args.use_train_test
    #     )

    n_clusters = args.n_clusters

    nce_embeddings, clust_head, outcode, q_code, actual = get_embeddings(learner, train_loader)

#     if (args.alpha == 0) and (args.beta == 1) and (args.gamma == 0):
    train_accuracy, train_nmi, train_ari = evaluation_helper(num_trials, n_clusters, actual, nce_embeddings, clust_head, outcode, q_code, use_kmeans=True)
#     else:
#         train_accuracy, train_nmi, train_ari = evaluation_helper(num_trials, n_clusters, actual, nce_embeddings, clust_head, outcode, q_code)

    nce_embeddings, clust_head, outcode, q_code, actual = get_embeddings(learner, test_loader)

#     if (args.alpha == 0) and (args.beta == 1) and (args.gamma == 0):
    test_accuracy, test_nmi, test_ari = evaluation_helper(num_trials, n_clusters, actual, nce_embeddings, clust_head, outcode, q_code, use_kmeans=True)
#     else:
#         test_accuracy, test_nmi, test_ari = evaluation_helper(num_trials, n_clusters, actual, nce_embeddings, clust_head, outcode, q_code)


    accuracy = {}
    nmi = {}
    ari = {}
    
    accuracy['train'], accuracy['test'] = train_accuracy, test_accuracy
    nmi['train'], nmi['test'] = train_nmi, test_nmi
    ari['train'], ari['test'] = train_ari, test_ari

    if return_embeddings:
        return accuracy, nmi, ari, ep, nce_embeddings, actual
    else:
        return accuracy, nmi, ari, ep
    

def write_csv(results_dict, file_name):
    df = pd.DataFrame(results_dict)
    with pd.ExcelWriter(file_name) as writer:

        for column in df.columns:
            train_test_df = pd.concat(
                [pd.DataFrame(df[column]['train']).drop('list', axis=0),pd.DataFrame(df[column]['test']).drop('list', axis=0)],
                axis=0, keys=['train', 'test']
            )
            train_test_df = train_test_df.transpose()
            train_test_df.to_excel(writer, sheet_name=column)            
    
