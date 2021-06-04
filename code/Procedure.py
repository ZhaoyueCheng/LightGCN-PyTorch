'''
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation
@author: Jianbai Ye (gusye@mail.ustc.edu.cn)

Design training and test process
'''
import world
import numpy as np
import torch
import utils
import dataloader
from pprint import pprint
from time import time
from tqdm import tqdm
import model
import multiprocessing
from sklearn.metrics import roc_auc_score


CORES = multiprocessing.cpu_count() // 2


def BPR_train_original(dataset, recommend_model, loss_class, epoch, sampler, w=None):
    Recmodel = recommend_model
    Recmodel.train()
    bpr: utils.BPRLoss = loss_class
    # allusers = list(range(dataset.n_users))
    # S, sam_time = utils.UniformSample_original(allusers, dataset)
    # print(f"BPR[sample time][{sam_time[0]:.1f}={sam_time[1]:.2f}+{sam_time[2]:.2f}]")
    # users = torch.Tensor(S[:, 0]).long()
    # posItems = torch.Tensor(S[:, 1]).long()
    # negItems = torch.Tensor(S[:, 2]).long()

    # users = users.to(world.device)
    # posItems = posItems.to(world.device)
    # negItems = negItems.to(world.device)

    # users, posItems, negItems = utils.shuffle(users,  posItems, negItems)
    total_batch = dataset.n_users // world.config['bpr_batch_size']
    aver_neg_loss = 0.
    aver_pos_loss = 0.
    aver_reg_loss = 0.
    for k in range(total_batch):
        samples = sampler.next_batch()
        S = samples[0]
        num_items_per_user = samples[1]

        users = torch.Tensor(S[:, 0]).long()
        posItems = torch.Tensor(S[:, 1]).long()
        negItems = torch.Tensor(S[:, 2:]).long()

        users = users.to(world.device)
        posItems = posItems.to(world.device)
        negItems = negItems.to(world.device)

        neg_loss, pos_loss, reg_loss = bpr.stageOne(users, posItems, negItems, num_items_per_user)
        aver_neg_loss += neg_loss
        aver_pos_loss += pos_loss
        aver_reg_loss += reg_loss
        if world.tensorboard:
            w.add_scalar(f'BPRNegLoss/BPR', neg_loss, epoch * int(len(users) / world.config['bpr_batch_size']) + k)
            w.add_scalar(f'BPRPosLoss/BPR', pos_loss, epoch * int(len(users) / world.config['bpr_batch_size']) + k)
            w.add_scalar(f'BPRRegLoss/BPR', reg_loss, epoch * int(len(users) / world.config['bpr_batch_size']) + k)

    aver_neg_loss = aver_neg_loss / total_batch
    aver_pos_loss = aver_pos_loss / total_batch
    aver_reg_loss = aver_reg_loss / total_batch
    return f"[BPR[aver neg loss{aver_neg_loss:.3e}, aver pos loss{aver_pos_loss:.3e}, aver reg loss{aver_reg_loss:.3e}]"
    
    
def test_one_batch(X):
    sorted_items = X[0].numpy()
    groundTrue = X[1]
    r = utils.getLabel(groundTrue, sorted_items)
    pre, recall, ndcg = [], [], []
    for k in world.topks:
        ret = utils.RecallPrecision_ATk(groundTrue, r, k)
        pre.append(ret['precision'])
        recall.append(ret['recall'])
        ndcg.append(utils.NDCGatK_r(groundTrue,r,k))
    return {'recall':np.array(recall), 
            'precision':np.array(pre), 
            'ndcg':np.array(ndcg)}
        
            
def Test(dataset, Recmodel, epoch, w=None, multicore=0):
    u_batch_size = world.config['test_u_batch_size']
    dataset: utils.BasicDataset
    testDict: dict = dataset.testDict
    Recmodel: model.LightGCN
    # eval mode with no dropout
    Recmodel = Recmodel.eval()
    max_K = max(world.topks)
    if multicore == 1:
        pool = multiprocessing.Pool(CORES)
    results = {'precision': np.zeros(len(world.topks)),
               'recall': np.zeros(len(world.topks)),
               'ndcg': np.zeros(len(world.topks))}
    with torch.no_grad():
        users = list(testDict.keys())
        try:
            assert u_batch_size <= len(users) / 10
        except AssertionError:
            print(f"test_u_batch_size is too big for this dataset, try a small one {len(users) // 10}")
        users_list = []
        rating_list = []
        groundTrue_list = []
        auc_record = []
        # ratings = []
        total_batch = len(users) // u_batch_size + 1
        for batch_users in utils.minibatch(users, batch_size=u_batch_size):
            allPos = dataset.getUserPosItems(batch_users)
            groundTrue = [testDict[u] for u in batch_users]
            batch_users_gpu = torch.Tensor(batch_users).long()
            batch_users_gpu = batch_users_gpu.to(world.device)

            rating = Recmodel.getUsersRating(batch_users_gpu)
            #rating = rating.cpu()
            exclude_index = []
            exclude_items = []
            for range_i, items in enumerate(allPos):
                exclude_index.extend([range_i] * len(items))
                exclude_items.extend(items)
            rating[exclude_index, exclude_items] = -(1<<10)
            _, rating_K = torch.topk(rating, k=max_K)
            rating = rating.cpu().numpy()
            # aucs = [
            #         utils.AUC(rating[i],
            #                   dataset,
            #                   test_data) for i, test_data in enumerate(groundTrue)
            #     ]
            # auc_record.extend(aucs)
            del rating
            users_list.append(batch_users)
            rating_list.append(rating_K.cpu())
            groundTrue_list.append(groundTrue)
        assert total_batch == len(users_list)
        X = zip(rating_list, groundTrue_list)
        if multicore == 1:
            pre_results = pool.map(test_one_batch, X)
        else:
            pre_results = []
            for x in X:
                pre_results.append(test_one_batch(x))
        scale = float(u_batch_size/len(users))
        for result in pre_results:
            results['recall'] += result['recall']
            results['precision'] += result['precision']
            results['ndcg'] += result['ndcg']
        results['recall'] /= float(len(users))
        results['precision'] /= float(len(users))
        results['ndcg'] /= float(len(users))
        # results['auc'] = np.mean(auc_record)
        if world.tensorboard:
            w.add_scalars(f'Test/Recall@{world.topks}',
                          {str(world.topks[i]): results['recall'][i] for i in range(len(world.topks))}, epoch)
            w.add_scalars(f'Test/Precision@{world.topks}',
                          {str(world.topks[i]): results['precision'][i] for i in range(len(world.topks))}, epoch)
            w.add_scalars(f'Test/NDCG@{world.topks}',
                          {str(world.topks[i]): results['ndcg'][i] for i in range(len(world.topks))}, epoch)
        if multicore == 1:
            pool.close()
        print(results)
        return results
