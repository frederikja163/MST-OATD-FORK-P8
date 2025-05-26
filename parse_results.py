import re
from datetime import datetime
from pprint import pprint
import json
import argparse
import glob
import os

def parse_results_log(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    blocks = []
    current_block = {}
    parsing_results = False

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Detect start of a new block
        if line.startswith("Timestamp:"):
            if current_block:
                blocks.append(current_block)
                current_block = {}
            current_block["hyperparameters"] = {}
            current_block["results"] = {}
            current_block["hyperparameters"]["timestamp"] = line.split(":", 1)[1].strip()
            parsing_results = False
        elif line.startswith("---------------------------------------------"):
            parsing_results = False
            continue
        elif re.match(r"^(PR_AUC|ROC_AUC|F1|PRECISION|RECALL|ACCURACY|THRESHOLD):", line):
            parsing_results = True

        # Parse hyperparameters or results
        if ":" in line and not line.startswith("Timestamp:") and not line.startswith("---"):
            key, val = map(str.strip, line.split(":", 1))

            # Try parsing the value into the right data type
            if re.match(r'^-?\d+\.\d+$', val):
                val = float(val)
            elif re.match(r'^-?\d+$', val):
                val = int(val)
            elif val.lower() in ["true", "false"]:
                val = val.lower() == "true"
            elif re.match(r"^20\d{2}-\d{2}-\d{2}", val):  # ISO timestamp
                val = datetime.fromisoformat(val)

            if parsing_results:
                current_block["results"][key.lower()] = val
            else:
                current_block["hyperparameters"][key.lower()] = val

    if current_block:
        blocks.append(current_block)

    return blocks

def load_all_results(files):
    all_blocks = []
    seen_checkpoints = set()

    for file_path in files:
        blocks = parse_results_log(file_path)
        for block in blocks:
            checkpoint_idx = block.get("hyperparameters", {}).get("checkpoint_idx")
            if checkpoint_idx is not None and checkpoint_idx not in seen_checkpoints:
                seen_checkpoints.add(checkpoint_idx)
                all_blocks.append(block)

    all_blocks.sort(key=lambda b: b.get("hyperparameters", {}).get("checkpoint_idx", float('inf')))
    return all_blocks

def create_latex_table(data):
    ratios = [0.5, 0.7, 1.0]
    grouped = {}

    # First pass: group entries
    for entry in data:
        hp = entry.get("hyperparameters", {})
        results = entry.get("results", {})
        dataset = hp.get("dataset")
        n_cluster = hp.get("n_cluster")
        lr_t = hp.get("lr_t")
        lr_s = hp.get("lr_s")
        s1_size = hp.get("s1_size")
        s2_size = hp.get("s2_size")
        obs_ratio = hp.get("obeserved_ratio")  # typo assumed intentional

        if None in (dataset, n_cluster, lr_t, lr_s, s1_size, s2_size, obs_ratio):
            continue

        key = (dataset, n_cluster, lr_t, lr_s, s1_size, s2_size)

        if key not in grouped:
            grouped[key] = {}

        grouped[key][obs_ratio] = results.get("pr_auc")

    # Filter to only complete entries (all 3 ratios)
    complete_rows = {
        key: ratio_map
        for key, ratio_map in grouped.items()
        if all(r in ratio_map for r in ratios)
    }

    # Compute min/max per ratio column
    ratio_values = {r: [] for r in ratios}
    for ratio_map in complete_rows.values():
        for r in ratios:
            ratio_values[r].append(ratio_map[r])

    ratio_min = {r: min(ratio_values[r]) for r in ratios}
    ratio_max = {r: max(ratio_values[r]) for r in ratios}

    # Second pass: build rows with formatting
    for key, ratio_map in complete_rows.items():
        dataset, n_cluster, lr_t, lr_s, s1_size, s2_size = key

        row_values = []
        for r in ratios:
            val = ratio_map[r]
            formatted = f"{val:.4f}"
            if val == ratio_max[r]:
                formatted = f"\\textbf{{{formatted}}}"
            elif val == ratio_min[r]:
                formatted = f"\\textit{{{formatted}}}"
            row_values.append(formatted)

        row = (
            f"\\textbf{{{dataset.capitalize()}}} & "
            f"{n_cluster} & {lr_t} & {lr_s} & "
            f"{s1_size} & {s2_size} & "
            f"{row_values[0]} & {row_values[1]} & {row_values[2]} \\\\"
        )
        print(row)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, required=True, help='directory containing results*.txt files')
    args = parser.parse_args()

    txt_files = glob.glob(os.path.join(args.dir, "results*.txt"))
    results = load_all_results(txt_files)
    # create_latex_table(results)

    with open(f"parsed_results_{datetime.today().strftime('%Y-%m-%d_%H-%M-%S')}.json", "w") as f:
        json.dump(results, f, indent=4, default=str)
