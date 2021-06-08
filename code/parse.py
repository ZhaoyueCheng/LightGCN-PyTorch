'''
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

@author: Jianbai Ye (gusye@mail.ustc.edu.cn)
'''
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Go lightGCN")

    parser.add_argument('--a_fold', type=int,default=100,
                        help="the fold num used to split large adj matrix, like gowalla")
    parser.add_argument('--testbatch', type=int,default=100,
                        help="the batch size of users for testing")
    parser.add_argument('--path', type=str,default="./checkpoints",
                        help="path to save weights")
    parser.add_argument('--topks', nargs='?',default="[5, 10, 20]",
                        help="@k test list")
    parser.add_argument('--tensorboard', type=int,default=1,
                        help="enable tensorboard")
    parser.add_argument('--comment', type=str,default="lgn")
    parser.add_argument('--load', type=int,default=0)
    parser.add_argument('--multicore', type=int, default=1, help='whether we use multiprocessing or not in test')
    parser.add_argument('--pretrain', type=int, default=0, help='whether we use pretrained weight or not')
    parser.add_argument('--seed', type=int, default=2020, help='random seed')
    parser.add_argument('--model', type=str, default='lgn', help='rec-model, support [mf, lgn]')

    parser.add_argument('--dropout', type=int,default=1,
                        help="using the dropout or not")
    parser.add_argument('--keepprob', type=float,default=0.8,
                        help="the batch size for bpr loss training procedure")

    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.001,
                        help="the learning rate")
    parser.add_argument('--bpr_batch', type=int, default=1000,
                        help="the batch size for metric loss training procedure")
    parser.add_argument('--recdim', type=int, default=64,
                        help="the embedding size of lightGCN")
    parser.add_argument('--layer', type=int, default=3,
                        help="the layer num of lightGCN")
    parser.add_argument('--dataset', type=str, default='TAFA-digital-music',
                        help="available datasets: [lastfm, gowalla, yelp2018, amazon]")
    parser.add_argument('--comb_method', type=str, default='sum',
                        help="combination method for combine the convolution layers [sum, mean, train, final]")
    
    # metric arguments
    parser.add_argument('--dist_method', type=str, default='L2', 
                        help="method of calculate distance: [L2, cos]")

    # MUL-Loss arguments
    parser.add_argument('--num_neg', type=int, default=10, help="number of negative edges")
    parser.add_argument('--margin', type=float, default=0.3, help="margin for the mul loss")
    parser.add_argument('--thresh', type=float, default=1.0, help="thresh for the mul loss")
    parser.add_argument('--alpha', type=float, default=1.0, help="pos, alpha for mul loss")
    parser.add_argument('--beta', type=float, default=5.0, help="neg, beta for mul loss")

    # Norm arguments
    # L2 norm
    parser.add_argument('--decay', type=float, default=1e-4,
                        help="the weight decay for l2 normalizaton")
    # clip norm
    parser.add_argument('--use_clip_norm', dest='use_clip_norm', action='store_true',
                        help="whether we use clip_norm")
    parser.set_defaults(use_clip_norm=False)
    parser.add_argument('--clip_norm', type=float, default=1.0,
                        help="clip_norm value")
    # keep on the surface of a unit sphere
    parser.add_argument('--use_fro_norm', dest='use_fro_norm', action='store_true',
                        help="whether we use fro_norm")
    parser.set_defaults(use_fro_norm=False)
    parser.add_argument('--fro_norm', type=float, default=1.0,
                        help="fro_norm value")

    return parser.parse_args()