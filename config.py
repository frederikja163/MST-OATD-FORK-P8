import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=800) # Default: 1600

parser.add_argument('--embedding_size', type=int, default=128) # Default 128
parser.add_argument('--hidden_size', type=int, default=512) # Default 512
parser.add_argument('--n_cluster', type=int, default=20) # Default 20

parser.add_argument('--pretrain_lr_s', type=float, default=2e-3) # Default 2e-3
parser.add_argument('--pretrain_lr_t', type=float, default=2e-3) # Default 2e-3

parser.add_argument('--lr_s', type=float, default=2e-4) # Default 2e-4
parser.add_argument('--lr_t', type=float, default=8e-5) # Default 8e-5

parser.add_argument('--epochs', type=int, default=5) # Default 5
parser.add_argument('--pretrain_epochs', type=int, default=6) # Default 6

parser.add_argument("--ratio", type=float, default=0.05, help="ratio of outliers") # Default 0.05
parser.add_argument("--distance", type=int, default=2) # Default 2
parser.add_argument("--fraction", type=float, default=0.2) # Default 0.2
parser.add_argument("--obeserved_ratio", type=float, default=1.0) # Default 1.0

parser.add_argument("--device", type=str, default='cuda') # Default cuda:0
parser.add_argument("--dataset", type=str, default='porto') # Default porto
parser.add_argument("--update_mode", type=str, default='pretrain') # Default pretrain

parser.add_argument("--train_num", type=int, default=80000)  # Default 80000

parser.add_argument("--s1_size", type=int, default=2) # Default 2 
parser.add_argument("--s2_size", type=int, default=4) # Default 4

parser.add_argument("--task", type=str, default='train') # Default train

args = parser.parse_args()

# python train.py --dataset porto --batch_size 1600 --pretrain_epochs 6
# python train.py --dataset cd --batch_size 300 --pretrain_epochs 3 --epochs 4
