import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--grid_size", type=float, default=0.1)
parser.add_argument("--dataset", type=str, default='porto')
parser.add_argument("--max_traj_time_delta", type=int, default=1900)
parser.add_argument("--find_boundary", type=bool, default=False)

parser.add_argument("--processes", type=int, default=10)
parser.add_argument("--chunk_size", type=int, default=800)

parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--epoch_split", type=float, default=0.5)


args = parser.parse_args()
