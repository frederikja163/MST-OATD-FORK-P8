# hyperparams.py
import itertools
import sys
import os

# Full hyperparameter ranges
N_CLUSTER = [10]
LR_S = [5e-05, 5.85e-05, 6.84e-05, 8e-05, 9.35e-05, 0.0001093, 0.0001278, 0.0001494, 0.0001746, 0.0002]
LR_T = [4e-05, 4.68e-05, 5.47e-05, 6.39e-05, 7.46e-05, 8.7e-05, 0.0001015, 0.0001183, 0.0001378, 0.00016]
S1_SIZE = [3]
S2_SIZE = [2]
DISTANCE = [2]
FRACTION = [0.2]
OBESERVED_RATIO = [0.7]

# Build full grid
full_grid = list(itertools.product(N_CLUSTER, LR_T, LR_S, S1_SIZE, S2_SIZE, DISTANCE, FRACTION, OBESERVED_RATIO))

# Define allowed indices (these must be indices from the full grid)
if os.path.exists("missing_indexes.txt"):
    with open("missing_indexes.txt", "r") as f:
        ALLOWED_INDICES = [int(line.strip()) for line in f if line.strip()]
else:
    ALLOWED_INDICES = []

# Filter grid to include only the allowed ones
filtered_grid = [full_grid[i] for i in ALLOWED_INDICES]

# Get the index from the filtered list
filtered_idx = int(sys.argv[1])
if filtered_idx < 0 or filtered_idx >= len(filtered_grid):
    raise IndexError(f"Filtered index {filtered_idx} out of range. Must be between 0 and {len(filtered_grid) - 1}")

# Get original index for checkpoint tracking
original_idx = ALLOWED_INDICES[filtered_idx]

# Unpack parameters
n_cluster, lr_t, lr_s, s1_size, s2_size, distance, fraction, obeserved_ratio = filtered_grid[filtered_idx]

# Output arguments with correct checkpoint index
print(f"""--n_cluster {n_cluster} --lr_t {lr_t} --lr_s {lr_s} --s1_size {s1_size} --s2_size {s2_size} 
          --distance {distance} --fraction {fraction} --obeserved_ratio {obeserved_ratio} --checkpoint_idx {original_idx}""")
