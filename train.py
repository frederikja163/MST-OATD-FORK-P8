import numpy as np
import torch
from torch.utils.data import DataLoader
import json

from config import args
from mst_oatd_trainer import train_mst_oatd, collate_fn, TrajectoryDataset



def main():
    train_trajs = np.load(f'./data/{args.dataset}/train_init.npy', allow_pickle=True)
    test_trajs = np.load(
        f'./data/{args.dataset}/outliers_data_init_{args.distance}_{args.fraction}_{args.obeserved_ratio}.npy', allow_pickle=True)
    outliers_idx = np.load(
        f"./data/{args.dataset}/outliers_idx_init_{args.distance}_{args.fraction}_{args.obeserved_ratio}.npy", allow_pickle=True)

    train_data = TrajectoryDataset(train_trajs)
    test_data = TrajectoryDataset(test_trajs)

    labels = np.zeros(len(test_data))
    for i in outliers_idx:
        labels[i] = 1

    train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn,
                              num_workers=8, pin_memory=True)
    outliers_loader = DataLoader(dataset=test_data, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn,
                                 num_workers=8, pin_memory=True)

    MST_OATD = train_mst_oatd(s_token_size, t_token_size, labels, train_loader, outliers_loader, time_interval, args)

    if args.task == 'train':

        MST_OATD.logger.info("Start pretraining!")

        for epoch in range(args.pretrain_epochs):
            MST_OATD.pretrain(epoch)

        MST_OATD.train_gmm()
        MST_OATD.save_weights_for_MSTOATD()

        MST_OATD.logger.info("Start training!")
        MST_OATD.load_mst_oatd()
        for epoch in range(args.epochs):
            MST_OATD.train(epoch)

    if args.task == 'test':
        MST_OATD.logger.info(f"Start testing with: d = {args.distance}, {chr(945)} = {args.fraction}, {chr(961)} = {args.obeserved_ratio}")

        checkpoint = torch.load(MST_OATD.path_checkpoint, weights_only=False)
        MST_OATD.MST_OATD_S.load_state_dict(checkpoint['model_state_dict_s'])
        MST_OATD.MST_OATD_T.load_state_dict(checkpoint['model_state_dict_t'])
        pr_auc = MST_OATD.detection()
        pr_auc = "%.4f" % pr_auc

        MST_OATD.logger.info(f"d = {args.distance}, {chr(945)} = {args.fraction}, {chr(961)} = {args.obeserved_ratio} produces PR_AUC: {pr_auc}")

    if args.task == 'train':
        MST_OATD.train_gmm_update()
        z = MST_OATD.get_hidden()
        MST_OATD.get_prob(z.cpu())


if __name__ == "__main__":
    with open(f'./data/{args.dataset}/metadata.json', 'r') as f:
        (lat_grid_num, lon_grid_num) = tuple(json.load(f))

    time_interval = 10
    num_days = 60

    if args.dataset == 'porto':
        #s_token_size = 51 * 119
        #t_token_size = 5760
        time_interval = 15
        num_days = 60

    elif args.dataset == 'cd':
        #s_token_size = 167 * 154
        #t_token_size = 8640
        time_interval = 10
        num_days = 60

    elif args.dataset == 'tdrive':
        time_interval = 600
        num_days = 10

    s_token_size = lat_grid_num * lon_grid_num
    seconds_a_day = 24*60*60
    t_token_size = (seconds_a_day // time_interval) * num_days

    main()