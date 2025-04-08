import os
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from config import args
from mst_oatd_trainer import train_mst_oatd, MyDataset, seed_torch, collate_fn

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def main(rank, world_size):
    setup(rank, world_size)

    if args.dataset == 'porto':
        s_token_size = 51 * 119
        t_token_size = 5760

    elif args.dataset == 'cd':
        s_token_size = 167 * 154
        t_token_size = 8640

    train_trajs = np.load('./data/{}/train_data_init.npy'.format(args.dataset), allow_pickle=True)
    test_trajs = np.load('./data/{}/outliers_data_init_{}_{}_{}.npy'.format(args.dataset, args.distance, args.fraction,
                                                                            args.obeserved_ratio), allow_pickle=True)
    outliers_idx = np.load("./data/{}/outliers_idx_init_{}_{}_{}.npy".format(args.dataset, args.distance, args.fraction,
                                                                             args.obeserved_ratio), allow_pickle=True)

    train_data = MyDataset(train_trajs)
    test_data = MyDataset(test_trajs)

    labels = np.zeros(len(test_trajs))
    for i in outliers_idx:
        labels[i] = 1

    train_sampler = DistributedSampler(train_data, num_replicas=world_size, rank=rank, shuffle=True)
    test_sampler = DistributedSampler(test_data, num_replicas=world_size, rank=rank, shuffle=False)

    train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, sampler=train_sampler, collate_fn=collate_fn,
                              num_workers=8, pin_memory=True)
    outliers_loader = DataLoader(dataset=test_data, batch_size=args.batch_size, sampler=test_sampler, collate_fn=collate_fn,
                                 num_workers=8, pin_memory=True)


    model = train_mst_oatd(s_token_size, t_token_size, labels, train_loader, outliers_loader, args)
    model = DDP(model, device_ids=[rank])

    if args.task == 'train':
        if rank == 0:
            model.logger.info("Start pretraining!")

        for epoch in range(args.pretrain_epochs):
            model.module.pretrain(epoch)

        model.module.train_gmm()
        model.module.save_weights_for_MSTOATD()

        if rank == 0:
            model.logger.info("Start training!")

        model.module.load_mst_oatd()
        for epoch in range(args.epochs):
            model.module.train(epoch)

    if args.task == 'test' and rank == 0:

        model.logger.info('Start testing!')
        model.logger.info("d = {}".format(args.distance) + ", " + chr(945) + " = {}".format(args.fraction) + ", "
              + chr(961) + " = {}".format(args.obeserved_ratio))

        checkpoint = torch.load(model.module.path_checkpoint, weights_only=False, map_location=f'cuda:{rank}')
        model.module.MST_OATD_S.load_state_dict(checkpoint['model_state_dict_s'])
        model.module.MST_OATD_T.load_state_dict(checkpoint['model_state_dict_t'])
        pr_auc = model.module.detection()
        pr_auc = "%.4f" % pr_auc
        model.logger.info("PR_AUC: {}".format(pr_auc))

    if args.task == 'train':
        model.module.train_gmm_update()
        z = model.module.get_hidden()
        model.module.get_prob(z.cpu())
    
    cleanup()


if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size,), nprocs=world_size, join=True)
