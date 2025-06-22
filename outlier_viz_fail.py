import folium
import pandas as pd
import ast
import numpy as np

# Load the data
df = pd.read_csv("outliers\outliers\outliers_20250523_110319.csv")  # Replace with your actual file path

# Function to parse tensor strings into coordinate arrays
def parse_tensor(tensor_str):
    try:
        # Remove 'tensor(' and ')' and any whitespace
        clean_str = tensor_str.replace('tensor(', '').replace(')', '').strip()
        # Convert to numpy array
        return np.array(ast.literal_eval(clean_str))
    except:
        return np.array([])  # Return empty array if parsing fails

# Parse the trajectory data and timestamps
df['coordinates'] = df['trajectory_data'].apply(parse_tensor)
df['timestamps'] = df['timestamps'].apply(parse_tensor)

# Create a base map (adjust center as needed)
map_center = [41.1579, -8.6291]  # Default to Porto coordinates
m = folium.Map(location=map_center, zoom_start=12, tiles='OpenStreetMap')

# Color coding for outliers
def get_color(row):
    if row['is_top_outlier']:
        return 'red'
    elif row['anomaly_score'] > 0.5:  # Adjust threshold as needed
        return 'orange'
    else:
        return 'blue'

# Add trajectories to the map
for idx, row in df.iterrows():
    coords = row['coordinates']
    times = row['timestamps']
    
    # Skip if no valid coordinates
    if len(coords) == 0:
        continue
        
    # Convert coordinates to [lat, lon] pairs
    # NOTE: You'll need to adjust this conversion based on how your coordinates are encoded
    # This is a placeholder - your actual coordinate conversion might be different
    points = [[i/10000, i/10000] for i in coords]  # Simple example scaling
    
    # Add the trajectory to the map
    folium.PolyLine(
        locations=points,
        color=get_color(row),
        weight=2 + (row['anomaly_score'] * 3),  # Thicker lines for higher anomaly scores
        opacity=0.7,
        tooltip=f"ID: {row['trajectory_idx']} | Score: {row['anomaly_score']:.2f}"
    ).add_to(m)

    # Add markers for start and end points if we have valid points
    if len(points) > 0:
        folium.CircleMarker(
            location=points[0],
            radius=3,
            color='green',
            fill=True,
            fill_color='green',
            fill_opacity=1,
            popup=f"Start of trajectory {row['trajectory_idx']}"
        ).add_to(m)
        
        folium.CircleMarker(
            location=points[-1],
            radius=3,
            color='purple',
            fill=True,
            fill_color='purple',
            fill_opacity=1,
            popup=f"End of trajectory {row['trajectory_idx']}"
        ).add_to(m)

# Save the map
m.save('trajectory_visualization.html')
m  # Display in Jupyter notebook if applicable