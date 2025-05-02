import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import re
import argparse
import numpy as np
from matplotlib import cm
from scipy.interpolate import griddata

def parse_log_file(file_path):
    with open(file_path, 'r') as f:
        content = f.read()

    entries = content.split('---------------------------------------------')
    results = []

    for entry in entries:
        result = {}
        for line in entry.strip().split('\n'):
            if ':' in line:
                key, value = map(str.strip, line.split(':', 1))
                key = key.lower().replace(" ", "_")
                try:
                    result[key] = float(value)
                except ValueError:
                    result[key] = value
        if result:
            results.append(result)
    return results


def make_3d_plot(data, param_x, param_y, metric_z):
    filtered_data = [
        d for d in data if param_x in d and param_y in d and metric_z in d
    ]

    print(f"\nFiltered {len(filtered_data)} valid entries.")
    for d in filtered_data:
        print(f"{param_x}={d[param_x]}, {param_y}={d[param_y]}, {metric_z}={d[metric_z]}")

    if not filtered_data:
        print(f"No data with all parameters: {param_x}, {param_y}, {metric_z}")
        return

    x = np.array([d[param_x] for d in filtered_data])
    y = np.array([d[param_y] for d in filtered_data])
    z = np.array([d[metric_z] for d in filtered_data])

    # Interpolate to a grid
    xi = np.linspace(min(x), max(x), 100)
    yi = np.linspace(min(y), max(y), 100)
    xi, yi = np.meshgrid(xi, yi)
    zi = griddata((x, y), z, (xi, yi), method='cubic')

    # Find the highest point on the surface
    max_z_index = np.unravel_index(np.argmax(zi), zi.shape)
    max_z_value = zi[max_z_index]
    max_x = xi[max_z_index]
    max_y = yi[max_z_index]

    # Plot the interpolated surface
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(xi, yi, zi, cmap=cm.viridis, edgecolor='none', antialiased=True)

    # Add a dot at the highest point
    ax.scatter(max_x, max_y, max_z_value, color='r', s=100, label='Max Point')

    fig.colorbar(surf, label=metric_z)
    ax.set_xlabel(param_x)
    ax.set_ylabel(param_y)
    ax.set_zlabel(metric_z)
    ax.set_title(f'{metric_z} by {param_x} and {param_y}')
    ax.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="3D plot from results log")
    parser.add_argument('--x', type=str, required=True, help='X-axis parameter (e.g. embedding_size)')
    parser.add_argument('--y', type=str, required=True, help='Y-axis parameter (e.g. hidden_size)')
    parser.add_argument('--z', type=str, required=True, help='Metric for Z-axis (e.g. roc_auc)')

    args = parser.parse_args()

    # Example usage
    file_path = 'results.txt'  # change this path
    data = parse_log_file(file_path)

    # Choose two parameters and one metric to plot
    make_3d_plot(data, param_x=args.x, param_y=args.y, metric_z=args.z)
