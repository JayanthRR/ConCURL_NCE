import torch
import torch.nn as nn
import time
import sys

softmax = nn.Softmax(dim=1).cuda()


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


def getQ(out_queue, epsilon=0.05):
    
    return distributed_sinkhorn(torch.exp(out_queue / epsilon).t(), 3)
    
def byol_loss_fn(x, y):
    #x = F.normalize(x, dim=-1, p=2)
    #y = F.normalize(y, dim=-1, p=2)
    return 2 - 2 * (x * y).sum(dim=-1)


def ByolLoss(features_one, features_two):
    online_pred_one = nn.functional.normalize(features_one['online_pred'], dim=1, p=2)
    online_pred_two = nn.functional.normalize(features_two['online_pred'], dim=1, p=2)
    target_proj_one = nn.functional.normalize(features_one['target_proj'], dim=1, p=2)
    target_proj_two = nn.functional.normalize(features_two['target_proj'], dim=1, p=2)

    byol_loss = byol_loss_fn(online_pred_one, target_proj_two).mean() + byol_loss_fn(online_pred_two, target_proj_one).mean()

    sys.stdout.flush()
    return byol_loss

def softSubLosses(outOne, outTwo,qOne, qTwo, param=0.1):
    pOne = softmax(outOne/param)
    pTwo = softmax(outTwo/param)
    subloss_1 = - torch.mean(torch.sum(qTwo * torch.log(pOne), dim=1))
    subloss_2 = - torch.mean(torch.sum(qOne * torch.log(pTwo), dim=1))
    return subloss_1, subloss_2


def SoftLoss(outcodes_one, outcodes_two, alpha=1, temperature=0.1, overclustering=False):
    if alpha > 0:
        if overclustering:
            out_one, out_two = outcodes_one['cTz_overcluster'], outcodes_two['cTz_overcluster']
        else:
            out_one, out_two = outcodes_one['cTz'], outcodes_two['cTz']
        #ATTENTION: I have deleted clone operations. Please think about it. My decision can be wrong!!!!
        with torch.no_grad():
            q_one = getQ(out_one)
            q_two = getQ(out_two)
        
        subloss_1, subloss_2 = softSubLosses(out_one, out_two, q_one, q_two, temperature)

        sys.stdout.flush()

        return (subloss_1 + subloss_2)/2.0, q_one, q_two
    else:
        return torch.tensor(0), None, None

def ConsensusLossForAGivenProjection(out_rand_one, out_rand_two, q_one, q_two, param=0.1):
    p_rand_one = softmax(out_rand_one/ param)
    p_rand_two = softmax(out_rand_two/ param)
    rand_loss_1 = -torch.mean(torch.sum(q_two * torch.log(p_rand_one), dim=1))
    rand_loss_2 = -torch.mean(torch.sum(q_one * torch.log(p_rand_two), dim=1))
    return (-torch.mean(torch.sum(q_two * torch.log(p_rand_one), dim=1)) - torch.mean(torch.sum(q_one * torch.log(p_rand_two), dim=1)))/2


def ConsensusLoss(gamma,  outcodes_one, outcodes_two, rand_outs_one, rand_outs_two, q_one, q_two, overclustering=False, temperature=0.1):
    loss = torch.tensor(0).cuda()

    if q_one is None or q_two is None:
        # check this when gamma>0 but alpha=0
        if overclustering:
            out_one, out_two = outcodes_one['cTz_overcluster'], outcodes_two['cTz_overcluster']
        else:
            out_one, out_two = outcodes_one['cTz'], outcodes_two['cTz']

        q_one = getQ(out_one)
        q_two = getQ(out_two)

    if gamma > 0:
        for randind in range(len(rand_outs_one)):

            if overclustering:
                temp = ConsensusLossForAGivenProjection(rand_outs_one[randind]['cTz_overcluster'], rand_outs_two[randind]['cTz_overcluster'], q_one, q_two, temperature)
                loss = loss + temp
            else:
                
                temp= ConsensusLossForAGivenProjection(rand_outs_one[randind]['cTz'], rand_outs_two[randind]['cTz'], q_one, q_two, temperature)
                loss = loss + temp

    sys.stdout.flush()

    return loss/len(rand_outs_one)
