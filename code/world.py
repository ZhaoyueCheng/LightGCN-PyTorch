'''
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

@author: Jianbai Ye (gusye@mail.ustc.edu.cn)
'''

import os
from os.path import join
import torch
from enum import Enum
from parse import parse_args
import multiprocessing

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
args = parse_args()

# ROOT_PATH = "/Users/gus/Desktop/light-gcn"
# ROOT_PATH = "/home/ubuntu/LightGCN-PyTorch/"
ROOT_PATH = "/home/zhaolin/research/LightGCN-PyTorch/"
CODE_PATH = join(ROOT_PATH, 'code')
DATA_PATH = join(ROOT_PATH, 'data')
BOARD_PATH = join(CODE_PATH, 'runs')
FILE_PATH = join(CODE_PATH, 'checkpoints')


if not os.path.exists(FILE_PATH):
    os.makedirs(FILE_PATH, exist_ok=True)


config = {}
all_dataset = ['lastfm', 'gowalla', 'yelp2018', 'amazon', 'amazon-music', 'amazon-grocery', 'ml']
all_models  = ['mf', 'lgn']
# config['batch_size'] = 4096
config['bpr_batch_size'] = args.bpr_batch
config['latent_dim_rec'] = args.recdim
config['lightGCN_n_layers']= args.layer
config['dropout'] = args.dropout
config['keep_prob']  = args.keepprob
config['A_n_fold'] = args.a_fold
config['test_u_batch_size'] = args.testbatch
config['multicore'] = args.multicore
config['lr'] = args.lr
config['pretrain'] = args.pretrain
config['A_split'] = False
config['bigdata'] = False
config['comb_method'] = args.comb_method

# metric config
config['dist_method'] = args.dist_method

# augmetation config
config['num_neg'] = args.num_neg
config['aug_method'] = args.aug_method
config['n_inner_pts'] = args.n_inner_pts
config['aug_norm'] = args.aug_norm
config['num_synthetic'] = args.num_synthetic

# loss config
config['margin'] = args.margin
config['loss'] = args.loss

# mul loss config
config['alpha'] = args.alpha
config['beta'] = args.beta
config['thresh'] = args.thresh

# norm config
config['decay'] = args.decay
config['use_clip_norm'] = args.use_clip_norm
config['clip_norm'] = args.clip_norm
config['use_fro_norm'] = args.use_fro_norm
config['fro_norm'] = args.fro_norm
config['norm_mode'] = args.norm_mode
config['norm_scale'] = args.norm_scale

GPU = torch.cuda.is_available()
device = torch.device('cuda' if GPU else "cpu")
CORES = multiprocessing.cpu_count() // 2
seed = args.seed

dataset = args.dataset
model_name = args.model
# if dataset not in all_dataset:
#     raise NotImplementedError(f"Haven't supported {dataset} yet!, try {all_dataset}")
if model_name not in all_models:
    raise NotImplementedError(f"Haven't supported {model_name} yet!, try {all_models}")

TRAIN_epochs = args.epochs
LOAD = args.load
PATH = args.path
topks = eval(args.topks)
tensorboard = args.tensorboard
comment = args.comment
# let pandas shut up
from warnings import simplefilter
simplefilter(action="ignore", category=FutureWarning)



def cprint(words : str):
    print(f"\033[0;30;43m{words}\033[0m")

logo = r"""
██╗      ██████╗ ███╗   ██╗
██║     ██╔════╝ ████╗  ██║
██║     ██║  ███╗██╔██╗ ██║
██║     ██║   ██║██║╚██╗██║
███████╗╚██████╔╝██║ ╚████║
╚══════╝ ╚═════╝ ╚═╝  ╚═══╝
"""
# font: ANSI Shadow
# refer to http://patorjk.com/software/taag/#p=display&f=ANSI%20Shadow&t=Sampling
# print(logo)