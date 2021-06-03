
import world
import utils
from world import cprint
import torch
import numpy as np
from tensorboardX import SummaryWriter
import time
import Procedure
from os.path import join

if __name__ == '__main__':
    utils.set_seed(world.seed)
    print(">>SEED:", world.seed)
    import register
    from register import dataset

    Recmodel = register.MODELS[world.model_name](world.config, dataset)
    Recmodel = Recmodel.to(world.device)
    # bpr = utils.BPRLoss(Recmodel, world.config)
    metric = utils.MetricLoss(Recmodel, world.config)

    weight_file = utils.getFileName()
    print(f"load and save to {weight_file}")
    if world.LOAD:
        try:
            Recmodel.load_state_dict(torch.load(weight_file,map_location=torch.device('cpu')))
            world.cprint(f"loaded model weights from {weight_file}") 
        except FileNotFoundError:
            print(f"{weight_file} not exists, start from beginning")

    # init tensorboard
    if world.tensorboard:
        w : SummaryWriter = SummaryWriter(
                                        join(world.BOARD_PATH, time.strftime("%m-%d-%Hh%Mm%Ss-") + "-" + world.comment)
                                        )
    else:
        w = None
        world.cprint("not enable tensorflowboard")
        
    # init sampler
    sampler = utils.WarpSampler(dataset, world.config['bpr_batch_size'], world.config['num_neg'])

    try:
        for epoch in range(world.TRAIN_epochs):
            print('======================')
            print(f'EPOCH[{epoch}/{world.TRAIN_epochs}]')
            start = time.time()
            if epoch % 10 == 0:
                cprint("[TEST]")
                Procedure.Test(dataset, Recmodel, epoch, w, world.config['multicore'])
            output_information = Procedure.Metric_train_original(dataset, Recmodel, metric, epoch, sampler,
                                                                neg_k=world.config['num_neg'], w=w)
            
            print(f'[saved][{output_information}]')
            torch.save(Recmodel.state_dict(), weight_file)
            print(f"[TOTAL TIME] {time.time() - start}")
        Procedure.Test(dataset, Recmodel, epoch, w, world.config['multicore'])
    finally:
        sampler.close()
        if world.tensorboard:
            w.close()