import os
import re

def extract_checkpoint_indices(directory="."):
    checkpoint_indices = set()

    for filename in os.listdir(directory):
        if filename.startswith("results") and filename.endswith(".txt"):
            with open(os.path.join(directory, filename), 'r') as file:
                content = file.read()
                matches = re.findall(r"checkpoint_idx:\s*(\d+)", content)
                checkpoint_indices.update(int(m) for m in matches)

    return checkpoint_indices

def find_missing_indices(used_indices, total_range=729):
    return sorted(set(range(total_range)) - used_indices)

if __name__ == "__main__":
    TOTAL = 729  # From 0 to 728
    used = extract_checkpoint_indices()
    missing = find_missing_indices(used, TOTAL)
    percentage_missing = (len(missing) / TOTAL) * 100

    print("Missing checkpoint indices:")
    print(missing)
    print(f"\nTotal missing: {len(missing)} out of {TOTAL}")
    print(f"Percentage missing: {percentage_missing:.2f}%")

    with open("missing_indexes.txt", "w") as f:
        for index in missing:
            f.write(f"{index}\n")
