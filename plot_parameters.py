import os
import re
import argparse
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

def extract_parameters(filepath):
    with open(filepath, "r") as f:
        content = f.read()

    values = {}
    for key, regex in PARAMETER_LABELS.items():
        match = re.search(regex, content)
        if match:
            values[key] = float(match.group(1))
    return values


def collect_all_parameters(log_dir):
    parameter_dicts = []
    for filename in os.listdir(log_dir):
        if filename.startswith("evaluation_metrics_") and filename.endswith(".txt"):
            filepath = os.path.join(log_dir, filename)
            extracted = extract_parameters(filepath)
            if len(extracted) >= 3:
                parameter_dicts.append(extracted)
    return parameter_dicts


def plot_3d_parameters(data, x_key, y_key, z_key, output_path):
    xs, ys, zs = [], [], []

    for entry in data:
        if x_key in entry and y_key in entry and z_key in entry:
            xs.append(entry[x_key])
            ys.append(entry[y_key])
            zs.append(entry[z_key])

    if not xs:
        print("No valid data points to plot.")
        return

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xs, ys, zs, c='blue', s=50)

    ax.set_xlabel(x_key.replace("_", " ").title())
    ax.set_ylabel(y_key.replace("_", " ").title())
    ax.set_zlabel(z_key.replace("_", " ").title())
    ax.set_title("3D Plot of Selected Parameters")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"3D plot saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="3D Parameter Plotter")
    parser.add_argument("dataset", help="Dataset name (folder under logs/)")
    parser.add_argument("--x", required=True, help="Parameter for X-axis")
    parser.add_argument("--y", required=True, help="Parameter for Y-axis")
    parser.add_argument("--z", required=True, help="Parameter for Z-axis")
    args = parser.parse_args()

    # Validate chosen parameters
    for param in [args.x, args.y, args.z]:
        if param not in PARAMETER_LABELS:
            print(f"Invalid parameter: {param}")
            print(f"Valid options: {list(PARAMETER_LABELS.keys())}")
            exit(1)

    log_dir = os.path.join("logs", args.dataset)
    if not os.path.exists(log_dir):
        print(f"Directory '{log_dir}' does not exist.")
        exit(1)

    data = collect_all_parameters(log_dir)
    output_path = os.path.join(log_dir, f"3d_plot_{args.x}_{args.y}_{args.z}.png")
    plot_3d_parameters(data, args.x, args.y, args.z, output_path)
