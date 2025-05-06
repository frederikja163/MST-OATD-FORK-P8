# hyperparams.py
import itertools
import sys

N_CLUSTER = [10, 20, 30]
LR_T = [4e-5, 8e-5, 1.6e-4]
LR_S = [1e-4, 2e-4, 4e-4]
S1_SIZE = [1, 2, 3]
S2_SIZE = [2, 4, 6]

DISTANCE = [2]
FRACTION = [0.2]
OBESERVED_RATIO = [0.5, 0.7, 1.0]

# Create the full grid
grid = list(itertools.product(N_CLUSTER, LR_T, LR_S, S1_SIZE, S2_SIZE, DISTANCE, FRACTION, OBESERVED_RATIO))

# Get the index from command-line argument
idx = int(sys.argv[1])
n_cluster, lr_t, lr_s, s1_size, s2_size, distance, fraction, obeserved_ratio = grid[idx]

# Print the args for the training script
print(f"""--n_cluster {n_cluster} --lr_t {lr_t} --lr_s {lr_s} --s1_size {s1_size} --s2_size {s2_size} 
          --distance {distance} --fraction {fraction} --obeserved_ratio {obeserved_ratio}""")
