"""
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

@author: Jianbai Ye (gusye@mail.ustc.edu.cn)

Define models here
"""
import world
import torch
from dataloader import BasicDataset
from torch import nn
import numpy as np


class BasicModel(nn.Module):    
    def __init__(self):
        super(BasicModel, self).__init__()
    
    def getUsersRating(self, users):
        raise NotImplementedError
    
class PairWiseModel(BasicModel):
    def __init__(self):
        super(PairWiseModel, self).__init__()
    def bpr_loss(self, users, pos, neg):
        """
        Parameters:
            users: users list 
            pos: positive items for corresponding users
            neg: negative items for corresponding users
        Return:
            (log-loss, l2-loss)
        """
        raise NotImplementedError
    
class PureMF(BasicModel):
    def __init__(self, 
                 config:dict, 
                 dataset:BasicDataset):
        super(PureMF, self).__init__()
        self.num_users  = dataset.n_users
        self.num_items  = dataset.m_items
        self.latent_dim = config['latent_dim_rec']
        self.f = nn.Sigmoid()
        self.__init_weight()
        
    def __init_weight(self):
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        print("using Normal distribution N(0,1) initialization for PureMF")
        
    def getUsersRating(self, users):
        users = users.long()
        users_emb = self.embedding_user(users)
        items_emb = self.embedding_item.weight
        scores = torch.matmul(users_emb, items_emb.t())
        return self.f(scores)
    
    def bpr_loss(self, users, pos, neg):
        users_emb = self.embedding_user(users.long())
        pos_emb   = self.embedding_item(pos.long())
        neg_emb   = self.embedding_item(neg.long())
        pos_scores= torch.sum(users_emb*pos_emb, dim=1)
        neg_scores= torch.sum(users_emb*neg_emb, dim=1)
        loss = torch.mean(nn.functional.softplus(neg_scores - pos_scores))
        reg_loss = (1/2)*(users_emb.norm(2).pow(2) + 
                          pos_emb.norm(2).pow(2) + 
                          neg_emb.norm(2).pow(2))/float(len(users))
        return loss, reg_loss
        
    def forward(self, users, items):
        users = users.long()
        items = items.long()
        users_emb = self.embedding_user(users)
        items_emb = self.embedding_item(items)
        scores = torch.sum(users_emb*items_emb, dim=1)
        return self.f(scores)

class LightGCN(BasicModel):
    def __init__(self, 
                 config:dict, 
                 dataset:BasicDataset):
        super(LightGCN, self).__init__()
        self.config = config
        self.dataset : dataloader.BasicDataset = dataset
        self.__init_weight()

    def __init_weight(self):
        self.num_users  = self.dataset.n_users
        self.num_items  = self.dataset.m_items
        self.latent_dim = self.config['latent_dim_rec']
        self.n_layers = self.config['lightGCN_n_layers']
        self.keep_prob = self.config['keep_prob']
        self.A_split = self.config['A_split']
        self.margin = self.config['margin']
        self.alpha = self.config['alpha']
        self.beta = self.config['beta']
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        if self.config['pretrain'] == 0:
            nn.init.xavier_uniform_(self.embedding_user.weight, gain=1)
            nn.init.xavier_uniform_(self.embedding_item.weight, gain=1)
            print('use xavier initilizer')
        else:
            self.embedding_user.weight.data.copy_(torch.from_numpy(self.config['user_emb']))
            self.embedding_item.weight.data.copy_(torch.from_numpy(self.config['item_emb']))
            print('use pretarined data')
        self.f = nn.Sigmoid()
        self.Graph = self.dataset.getSparseGraph()
        print(f"lgn is already to go(dropout:{self.config['dropout']})")

        # print("save_txt")
    def __dropout_x(self, x, keep_prob):
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index]/keep_prob
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g
    
    def __dropout(self, keep_prob):
        if self.A_split:
            graph = []
            for g in self.Graph:
                graph.append(self.__dropout_x(g, keep_prob))
        else:
            graph = self.__dropout_x(self.Graph, keep_prob)
        return graph
    
    def computer(self):
        """
        propagate methods for lightGCN
        """       
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])
        #   torch.split(all_emb , [self.num_users, self.num_items])
        embs = [all_emb]
        if self.config['dropout']:
            if self.training:
                print("droping")
                g_droped = self.__dropout(self.keep_prob)
            else:
                g_droped = self.Graph        
        else:
            g_droped = self.Graph    
        
        for layer in range(self.n_layers):
            if self.A_split:
                temp_emb = []
                for f in range(len(g_droped)):
                    temp_emb.append(torch.sparse.mm(g_droped[f], all_emb))
                side_emb = torch.cat(temp_emb, dim=0)
                all_emb = side_emb
            else:
                all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        #print(embs.size())
        light_out = torch.mean(embs, dim=1)
        # light_out = torch.sum(embs, dim=1)
        # light_out = embs.view(embs.shape[0], -1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items
    
    def getUsersRating(self, users):
        all_users, all_items = self.computer()
        users_emb = all_users[users.long()].unsqueeze(1)
        items_emb = all_items.unsqueeze(0)
        rating = -torch.sum((users_emb - items_emb) ** 2, 2)
        return rating
    
    def getEmbedding(self, users, pos_items, neg_items):
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego
    
    # def bpr_loss(self, users, pos, neg):
    #     (users_emb, pos_emb, neg_emb,
    #     userEmb0,  posEmb0, negEmb0) = self.getEmbedding(users.long(), pos.long(), neg.long())
    #     reg_loss = (1/2)*(userEmb0.norm(2).pow(2) +
    #                      posEmb0.norm(2).pow(2)  +
    #                      negEmb0.norm(2).pow(2))/float(len(users))
    #     pos_scores = torch.mul(users_emb, pos_emb)
    #     pos_scores = torch.sum(pos_scores, dim=1)
    #     neg_scores = torch.mul(users_emb, neg_emb)
    #     neg_scores = torch.sum(neg_scores, dim=1)
    #
    #     loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))
    #
    #     return loss, reg_loss

    def metric_loss(self, S, num_items_per_user):
        users = torch.Tensor(S[:, 0]).long()
        pos_items = torch.Tensor(S[:, 1]).long()
        neg_items = torch.Tensor(S[:, 2:]).long()

        users = users.to(world.device)
        pos_items = pos_items.to(world.device)
        neg_items = neg_items.to(world.device)

        (users_emb, pos_emb, neg_emb,
         userEmb0, posEmb0, negEmb0) = self.getEmbedding(users, pos_items, neg_items)
        
        reg_loss = (1 / 2) * (userEmb0.norm(2).pow(2) +
                              posEmb0.norm(2).pow(2) +
                              negEmb0.norm(2).pow(2)) / float(len(users))

        # positive item to user distance (N)
        pos_distances = torch.sum((users_emb - pos_emb) ** 2, 1)
        # distance to negative items (N x W)
        distance_to_neg_items = torch.sum((users_emb.unsqueeze(-1) - neg_emb.transpose(-2, -1)) ** 2, 1)

        start_idx = 0
        pos_lengths = []
        neg_length = []
        for i in num_items_per_user:

            max_pos_length = pos_distances[start_idx: start_idx+i].max()
            pos_lengths.append(max_pos_length)
            
            min_neg_length = distance_to_neg_items[start_idx: start_idx+i].min()
            neg_length.append(min_neg_length)
            
            start_idx += i

        num_items_per_user = torch.LongTensor(num_items_per_user)

         # negative mining using max pos length
        pos_lengths = torch.repeat_interleave(torch.tensor(pos_lengths), num_items_per_user)
        if torch.cuda.is_available():
            pos_lengths = pos_lengths.to(world.device)
        neg_idx = (distance_to_neg_items - (self.margin + pos_lengths.unsqueeze(-1))) >= 0
        distance_to_neg_items = distance_to_neg_items + torch.where(neg_idx, float('inf'), 0.)
        neg_loss = 1.0 / self.beta * torch.log(1 + torch.sum(torch.exp(-self.beta * (distance_to_neg_items + self.margin)))).sum()

        # positive mining using min neg length
        neg_length = torch.repeat_interleave(torch.tensor(neg_length), num_items_per_user)
        if torch.cuda.is_available():
            neg_length = neg_length.to(world.device)
        pos_idx = (pos_distances - (neg_length - self.margin)) <= 0
        pos_distances = pos_distances + torch.where(pos_idx, -float('inf'), 0.)
        pos_loss = 1.0 / self.alpha * torch.log(1 + torch.sum(torch.exp(self.alpha * (pos_distances + self.margin)))).sum()
        
        # print('neg_total:{:.2f} pos_total:{:.2f}'.format(neg_loss, pos_loss))

        return neg_loss+pos_loss, reg_loss


    # def metric_loss(self, users, pos, neg):
    #     (users_emb, pos_emb, neg_emb,
    #      userEmb0, posEmb0, negEmb0) = self.getEmbedding(users.long(), pos.long(), neg.long())
    #     reg_loss = (1 / 2) * (userEmb0.norm(2).pow(2) +
    #                           posEmb0.norm(2).pow(2) +
    #                           negEmb0.norm(2).pow(2)) / float(len(users))
    #     embedding_aug = True
    #     if embedding_aug:
    #         neg_items = self.get_embedding_aug(neg_emb, 5, False, 5)
    #     else:
    #         neg_items = neg_emb

    #     pos_distances = torch.sum((users_emb - pos_emb) ** 2, 1)
    #     neg_distances = torch.sum((users_emb.unsqueeze(-2) - neg_items) ** 2, -1)
    #     closest_negative_item_distances = neg_distances.min(1)[0]

    #     distance = pos_distances - closest_negative_item_distances + self.margin
    #     loss = torch.nn.functional.relu(distance)

    #     return loss, reg_loss

    def clip_norm_op(self, clip_norm=1):
        norm_user = (self.embedding_user.weight.data ** 2).sum(-1, keepdim=True)
        self.embedding_user.weight.data = torch.where(norm_user < clip_norm ** 2,
                                                      self.embedding_user.weight.data,
                                                      self.embedding_user.weight.data * clip_norm / (norm_user + 1e-12))

        norm_item = (self.embedding_item.weight.data ** 2).sum(-1, keepdim=True)
        self.embedding_item.weight.data = torch.where(norm_item < clip_norm ** 2,
                                                      self.embedding_item.weight.data,
                                                      self.embedding_item.weight.data * clip_norm / (norm_item + 1e-12))

        # self.embedding_user.weight.data = torch.nn.functional.normalize(self.embedding_user.weight.data, 2, -1)
        # self.embedding_item.weight.data = torch.nn.functional.normalize(self.embedding_item.weight.data, 2, -1)

    def normalize_op(self):
        self.embedding_user.weight.data = torch.nn.functional.normalize(self.embedding_user.weight.data, 2, -1)
        self.embedding_item.weight.data = torch.nn.functional.normalize(self.embedding_item.weight.data, 2, -1)
       
    # def forward(self, users, items):
    #     # compute embedding
    #     all_users, all_items = self.computer()
    #     # print('forward')
    #     #all_users, all_items = self.computer()
    #     users_emb = all_users[users]
    #     items_emb = all_items[items]
    #     inner_pro = torch.mul(users_emb, items_emb)
    #     gamma     = torch.sum(inner_pro, dim=1)
    #     return gamma

    def get_embedding_aug(self, embeddings, n_inner_pts=5, normalize=True, num_synthetic=5):

        # N = batch size,
        # K = embedding size,
        # W = number of negative samples per a user-positive-item pair

        # negative item embedding (N, W, K)
        num_neg = embeddings.shape[1]

        import itertools
        all_combinations = list(itertools.combinations(range(num_neg), 2))
        import random
        random.shuffle(all_combinations)
        all_combinations = all_combinations[:num_synthetic]

        axes_point_0 = [x[0] for x in all_combinations]
        axes_point_1 = [x[1] for x in all_combinations]

        axes_embeddings_0 = embeddings.clone()[:, axes_point_0, :]
        axes_embeddings_1 = embeddings.clone()[:, axes_point_1, :]

        concat_embeddings = embeddings.clone()

        total_length = float(n_inner_pts + 1)

        for n_idx in range(n_inner_pts):
            left_length = float(n_idx + 1)
            right_length = total_length - left_length

            inner_pts = (axes_embeddings_0 * left_length + axes_embeddings_1 * right_length) / total_length

            if normalize:
                ## normalize by clipping normalize
                # inner_pts = clip_by_norm(inner_pts, self.clip_norm)
                ## normalize by standard normalize function
                inner_pts = torch.nn.functional.normalize(inner_pts, 2, -1)

                # # test normalize again with original norm
                # original_norm = (torch.norm(axes_embeddings_0, dim=-1) * left_length +
                #                  torch.norm(axes_embeddings_1, dim=-1) * right_length) / total_length
                # inner_pts = inner_pts * original_norm.unsqueeze(-1)

            concat_embeddings = torch.cat((concat_embeddings, inner_pts), dim=1)

        return concat_embeddings