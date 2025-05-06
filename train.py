import numpy as np
import torch
from torch.utils.data import DataLoader

from config import args
from mst_oatd_trainer import train_mst_oatd, MyDataset, seed_torch, collate_fn
from datetime import datetime
import os

def main():
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
    labels = labels

    train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn,
                              num_workers=8, pin_memory=True)
    outliers_loader = DataLoader(dataset=test_data, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn,
                                 num_workers=8, pin_memory=True)

    MST_OATD = train_mst_oatd(s_token_size, t_token_size, labels, train_loader, outliers_loader, args)

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

        MST_OATD.logger.info('Start testing!')
        MST_OATD.logger.info("d = {}".format(args.distance) + ", " + chr(945) + " = {}".format(args.fraction) + ", "
              + chr(961) + " = {}".format(args.obeserved_ratio))

        checkpoint = torch.load(MST_OATD.path_checkpoint, weights_only=False)
        MST_OATD.MST_OATD_S.load_state_dict(checkpoint['model_state_dict_s'])
        MST_OATD.MST_OATD_T.load_state_dict(checkpoint['model_state_dict_t'])
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

        output_path = os.path.join(os.getcwd(), f"results_{args.dataset}_{datetime.today().strftime('%Y-%m-%d')}.txt")

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


if __name__ == "__main__":

    if args.dataset == 'porto':
        s_token_size = 51 * 119
        t_token_size = 5760

    elif args.dataset == 'cd':
        s_token_size = 167 * 154
        t_token_size = 8640

    main()
