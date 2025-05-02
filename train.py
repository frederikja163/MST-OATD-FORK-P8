import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist
import torch.multiprocessing as mp
from config import args
from mst_oatd_trainer import train_mst_oatd, MyDataset, seed_torch, collate_fn
import os
import time
from datetime import datetime

# Set environment variables for distributed training
os.environ['MASTER_ADDR'] = '127.0.0.1'  # IP address of the master node
os.environ['MASTER_PORT'] = '29500'      # Open port for communication
os.environ['RANK'] = '0'                 # Rank of the current process
os.environ['WORLD_SIZE'] = '4'           # Total number of processes (GPUs in this case)
os.environ['LOCAL_RANK'] = '0'


def main_worker(rank, world_size):
    try:

        # Initialize the process group for distributed training
        dist.init_process_group(backend='nccl', world_size=world_size, rank=rank)

        if args.dataset == 'porto':
            args.s_token_size = 51 * 119
            args.t_token_size = 5760
        elif args.dataset == 'cd':
            args.s_token_size = 167 * 154
            args.t_token_size = 8640

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.cuda.set_device(rank)  # Set the current device to the rank of the process

        # Load data
        train_trajs = np.load(f'./data/{args.dataset}/train_data_init.npy', allow_pickle=True)
        test_trajs = np.load(f'./data/{args.dataset}/outliers_data_init_{args.distance}_{args.fraction}_{args.obeserved_ratio}.npy', allow_pickle=True)
        outliers_idx = np.load(f"./data/{args.dataset}/outliers_idx_init_{args.distance}_{args.fraction}_{args.obeserved_ratio}.npy", allow_pickle=True)

        train_data = MyDataset(train_trajs)
        test_data = MyDataset(test_trajs)

        labels = np.zeros(len(test_trajs))
        for i in outliers_idx:
            labels[i] = 1

        num_workers = min(0, os.cpu_count()) 

        # Create samplers for training and testing
        train_sampler = DistributedSampler(train_data, num_replicas=world_size, rank=rank)
        outlier_sampler = DistributedSampler(test_data, num_replicas=world_size, rank=rank)

        # DataLoader setup
        train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn,
                                num_workers=num_workers, pin_memory=True, sampler=train_sampler)
        outliers_loader = DataLoader(dataset=test_data, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn,
                                    num_workers=num_workers, pin_memory=True, sampler=outlier_sampler)

        # Initialize model
        MST_OATD = train_mst_oatd(args.s_token_size, args.t_token_size, labels, train_loader, outliers_loader, args, rank=rank)
        MST_OATD.logger.info(f"Using device: {device}!")
        MST_OATD.logger.info(f"Number of workers (CPU): {num_workers}!")

        # Move model to device
        MST_OATD.MST_OATD_S.to(device)
        MST_OATD.MST_OATD_T.to(device)

        # Wrap models with DistributedDataParallel
        MST_OATD.MST_OATD_S = nn.parallel.DistributedDataParallel(MST_OATD.MST_OATD_S, device_ids=[rank], output_device=rank)
        MST_OATD.MST_OATD_T = nn.parallel.DistributedDataParallel(MST_OATD.MST_OATD_T, device_ids=[rank], output_device=rank)

        if args.task == 'train':
            start_time = time.time()
            MST_OATD.logger.info(f"Start pretraining! Epochs Count: {args.pretrain_epochs}")
            for epoch in range(args.pretrain_epochs):
                MST_OATD.pretrain(epoch)
            MST_OATD.logger.info("--- %s seconds for pretrain ---" % round(time.time() - start_time, 2))

            MST_OATD.train_gmm()
            MST_OATD.save_weights_for_MSTOATD()

            MST_OATD.logger.info("Start training!")
            MST_OATD.load_mst_oatd()

            start_time = time.time()
            for epoch in range(args.epochs):
                MST_OATD.train(epoch)
            MST_OATD.logger.info("--- %s seconds for train ---" % round(time.time() - start_time, 2))

        if args.task == 'test':

            MST_OATD.logger.info('Start testing!')

            checkpoint = torch.load(MST_OATD.path_checkpoint, map_location=device)

            MST_OATD.MST_OATD_S.module.load_state_dict(checkpoint['model_state_dict_s'])
            MST_OATD.MST_OATD_T.module.load_state_dict(checkpoint['model_state_dict_t'])

            metrics = MST_OATD.detection()

            # Format metrics to display
            formatted_metrics = "\n".join([f"{k.upper()}: {metrics[k]:.4f}" for k in metrics])

            # Log to console
            MST_OATD.logger.info("Evaluation Metrics:")
            for k, v in metrics.items():
                MST_OATD.logger.info(f"{k.upper()}: {v:.4f}")

            # Build the output string with timestamp, args, and metrics
            output_string = f"Timestamp: {datetime.now()}\n"

            # Add all arguments dynamically
            for arg, value in vars(args).items():
                output_string += f"{arg}: {value}\n"

            # Add the formatted metrics to the output string
            output_string += f"\n{formatted_metrics}\n"
            output_string += "---------------------------------------------\n"

            output_path = os.path.join(os.getcwd(), 'results.txt')

            # Check if the file exists; if not, create it
            if not os.path.exists(output_path):
                with open(output_path, 'w') as f:
                    f.write("Results Log\n")
                    f.write("===========\n")
                    f.write("Timestamp, Batch Size, Embedding Size, Hidden Size, N Cluster, Pretrain LR S, Pretrain LR T, "
                            "LR S, LR T, Epochs, Pretrain Epochs, Ratio, Distance, Fraction, Observed Ratio, Device, "
                            "Dataset, Update Mode, Train Num, S1 Size, S2 Size, Task, AUC, Precision, Recall, F1, Accuracy\n")
                    f.write("====================================================================================================\n")

            with open(output_path, 'a') as f:
                f.write(output_string)

        if args.task == 'train':
            MST_OATD.train_gmm_update()
            z = MST_OATD.get_hidden()
            MST_OATD.get_prob(z.cpu())

        MST_OATD.logger.info("Finished!")

    finally:
        dist.destroy_process_group()


def main():
    world_size = torch.cuda.device_count()  # Number of GPUs
    mp.spawn(main_worker, nprocs=world_size, args=(world_size,))


if __name__ == "__main__":
    main()
