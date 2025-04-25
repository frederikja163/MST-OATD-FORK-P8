# hyperparams.py
import itertools
import sys

# FRACTIONS = [0.1, 0.2, 0.3]
# OBESERVED_RATIOS = [0.5, 0.75, 1.0]
# DISTANCES = [1, 2]
FRACTIONS = [0.2]
OBESERVED_RATIOS = [1.0]
DISTANCES = [2]

# Create the full grid
grid = list(itertools.product(FRACTIONS, OBESERVED_RATIOS, DISTANCES))

# Get the index from command-line argument
idx = int(sys.argv[1])
fraction, obeserved_ratio, distance = grid[idx]

# Print the args for the training script
print(f"--fraction {fraction} --obeserved_ratio {obeserved_ratio} --distance {distance}")
