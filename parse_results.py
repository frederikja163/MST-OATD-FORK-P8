import re
from datetime import datetime
from pprint import pprint
import json
import argparse

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

    # Don't forget to append the last block
    if current_block:
        blocks.append(current_block)

    return blocks

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, required=True, help='the path to the file you want to parse')

    args = parser.parse_args()
    results = parse_results_log(args.file)

    # Save to JSON file
    with open(f"parsed_results_{datetime.today().strftime('%Y-%m-%d_%H-%M-%S')}.json", "w") as f:
        json.dump(results, f, indent=4, default=str)