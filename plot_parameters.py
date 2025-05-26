import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse
import numpy as np
from matplotlib import cm
import json
from collections import defaultdict
from scipy.interpolate import griddata
import pandas as pd

def parse_json_file(file_path):
    with open(file_path, 'r') as f:
        content = json.load(f)

    results = []
    for entry in content:
        result = {}
        result.update(entry.get("hyperparameters", {}))
        result.update(entry.get("results", {}))
        for key, value in result.items():
            if isinstance(value, str):
                try:
                    result[key] = float(value)
                except ValueError:
                    pass
        results.append(result)
    return results


def make_heatmap(data, param_x, param_y, metric_z, hyperparameter_filter=None):
    # Filter out entries missing any of the required keys
    filtered = [
        d for d in data 
        if param_x in d and param_y in d and metric_z in d
    ]

    if hyperparameter_filter is not None:
        filtered = [
            d for d in filtered 
            if all(d.get(key) == value for key, value in hyperparameter_filter.items())
        ]
    
    filtered = [d for d in filtered if d[metric_z] >= 0.74]

    # Create DataFrame from filtered data
    df = pd.DataFrame(filtered)[[param_x, param_y, metric_z]]
    
    heatmap_data = df.pivot_table(
        index=param_y,
        columns=param_x,
        values=metric_z,
        aggfunc='mean'  # or 'max', 'min', etc. depending on your goal
    )

    # Mask NaNs for custom coloring
    masked_data = np.ma.masked_invalid(heatmap_data.values)

    # heatmap_data = heatmap_data.dropna(axis=0, how='any')  # drop rows with any NaN
    # heatmap_data = heatmap_data.dropna(axis=1, how='any')  # drop columns with any NaN
    
    # Sort index and columns for nice axis ordering if numeric
    try:
        heatmap_data = heatmap_data.sort_index(axis=0)
        heatmap_data = heatmap_data.sort_index(axis=1)
    except Exception:
        pass
    
    plt.figure(figsize=(8,6))
    cmap = plt.get_cmap('gray')
    cmap.set_bad(color='black')  # color for NaNs
    im = plt.imshow(masked_data, cmap=cmap, interpolation='nearest', aspect='auto')
    
    plt.xlabel(param_x)
    plt.ylabel(param_y)
    filter_text = f" (Filter: {hyperparameter_filter})" if hyperparameter_filter else ""
    # plt.title(f'Max {metric_z} by {param_x} and {param_y}{filter_text}')
    
    plt.xticks(ticks=np.arange(len(heatmap_data.columns)), labels=heatmap_data.columns, rotation=45)
    plt.yticks(ticks=np.arange(len(heatmap_data.index)), labels=heatmap_data.index)
    
    cbar = plt.colorbar(im)
    cbar.set_label(metric_z)
    
    plt.tight_layout()
    plt.show()


def make_3d_plot(data, param_x, param_y, metric_z, interpolated=False, grid_points=100, method='cubic', hyperparameter_filter=None):
    """
    Parameters:
    - data: List of dictionaries containing parameter values
    - param_x: Name of x-axis parameter
    - param_y: Name of y-axis parameter
    - metric_z: Name of z-axis metric
    - interpolated: Boolean to toggle interpolation (False for raw triangulation)
    - grid_points: Resolution for interpolation grid (if interpolated=True)
    - method: Interpolation method ('linear', 'cubic', 'nearest')
    - hyperparameter_filter: Dictionary of {hyperparameter: value} to filter specific parameter combinations
    """
    # Initial filter for required parameters
    filtered_data = [d for d in data if param_x in d and param_y in d and metric_z in d]
    
    # Apply hyperparameter filter if specified
    if hyperparameter_filter is not None:
        filtered_data = [
            d for d in filtered_data 
            if all(d.get(key) == value for key, value in hyperparameter_filter.items())
        ]
    
    # Filter out data points where metric_z < 0.5
    filtered_data = [d for d in filtered_data if d[metric_z] >= 0.5]

    if not filtered_data:
        required_params = [param_x, param_y, metric_z]
        if hyperparameter_filter:
            required_params += list(hyperparameter_filter.keys())
        print(f"No data with all required parameters: {required_params}")
        return

    # Keep only the highest z for each (x,y) pair
    max_points = defaultdict(float)
    coord_map = {}
    
    for d in filtered_data:
        x_val = d[param_x]
        y_val = d[param_y]
        z_val = d[metric_z]
        key = (x_val, y_val)
        
        if key not in max_points or z_val > max_points[key]:
            max_points[key] = z_val
            coord_map[key] = (x_val, y_val, z_val)

    # Extract unique max points
    unique_points = list(coord_map.values())
    x = np.array([p[0] for p in unique_points])
    y = np.array([p[1] for p in unique_points])
    z = np.array([p[2] for p in unique_points])

    # Find global maximum (from raw data)
    max_idx = np.argmax(z)
    max_x, max_y, max_z = x[max_idx], y[max_idx], z[max_idx]

    # Plotting
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    if interpolated:
        # --- INTERPOLATED MODE ---
        # Create grid and interpolate
        xi = np.linspace(min(x), max(x), grid_points)
        yi = np.linspace(min(y), max(y), grid_points)
        xi, yi = np.meshgrid(xi, yi)
        zi = griddata((x, y), z, (xi, yi), method=method)
        
        # Plot surface
        surf = ax.plot_surface(xi, yi, zi, cmap=cm.viridis, 
                              rstride=1, cstride=1, alpha=0.8,
                              linewidth=0, antialiased=True)
    else:
        # --- RAW TRIANGULATED MODE ---
        surf = ax.plot_trisurf(x, y, z, cmap=cm.viridis, 
                              edgecolor='grey', linewidth=0.2, 
                              alpha=0.8, antialiased=True)

    # Mark maximum point (from raw data, not interpolation)
    ax.scatter([max_x], [max_y], [max_z], 
               color='r', s=200, label=f'Max {metric_z} = {max_z:.2f}',
               depthshade=False)

    # Add color bar and labels
    fig.colorbar(surf, label=metric_z)
    ax.set_xlabel(param_x)
    ax.set_ylabel(param_y)
    ax.set_zlabel(metric_z)
    
    title_method = "Interpolated" if interpolated else "Triangulated"
    filter_text = f" (Filter: {hyperparameter_filter})" if hyperparameter_filter else ""
    # ax.set_title(f'{title_method} Max {metric_z} by {param_x} and {param_y}{filter_text}')
    ax.legend()

    plt.tight_layout()
    plt.show()
                             

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="3D plot from JSON results")
    parser.add_argument('--x', type=str, required=True, help='X-axis parameter (e.g. embedding_size)')
    parser.add_argument('--y', type=str, required=True, help='Y-axis parameter (e.g. hidden_size)')
    parser.add_argument('--z', type=str, required=True, help='Metric for Z-axis (e.g. roc_auc)')
    parser.add_argument('--file', type=str, default='results.json', help='Path to the JSON file')

    args = parser.parse_args()

    data = parse_json_file(args.file)
    # make_3d_plot(data, param_x=args.x, param_y=args.y, metric_z=args.z, hyperparameter_filter={"obeserved_ratio": 1.0})
    make_heatmap(data, param_x=args.x, param_y=args.y, metric_z=args.z, hyperparameter_filter={"obeserved_ratio": 0.7})
