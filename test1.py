import folium
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Point, LineString

# data = np.load("data/porto/outliers_data_1_2_0.1_0.5.npy", allow_pickle=True)
# data = np.load("datasets\porto\porto.csv", allow_pickle=True)

# # Convert to DataFrame - first inspect the actual structure
# print("Raw data shape:", data.shape)
# print("First row:", data[0])
# exit()
# # Assuming your data has coordinates in columns 0 (longitude) and 1 (latitude)
# # Adjust these indices based on your actual data structure
# df = pd.DataFrame(data)
# df.columns = ['Longitude', 'Latitude', 'Name']  # Adjust column names as needed

df = pd.read_csv("datasets\porto\porto.csv")

# Create DataFrame and GeoDataFrame
geometry = [Point(xy) for xy in zip(df['Longitude'], df['Latitude'])]
gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")

# Create a route (trajectory) connecting the points in order
route = LineString(zip(df['Longitude'], df['Latitude']))

# Create the base map centered on Porto
porto_coords = [41.1579, -8.6291]
m = folium.Map(location=porto_coords, zoom_start=14, tiles='OpenStreetMap')

# Add the trajectory (route) to the map
folium.PolyLine(
    locations=[[lat, lon] for lat, lon in zip(df['Latitude'], df['Longitude'])],
    color='blue',
    weight=5,
    opacity=0.7,
    tooltip="Suggested Route"
).add_to(m)

# Add markers for each point
for idx, row in df.iterrows():
    folium.CircleMarker(
        location=[row['Latitude'], row['Longitude']],
        radius=5,          # Size of the dot
        color='red',       # Border color
        fill=True,
        fill_color='red',  # Inner color
        fill_opacity=1,
        popup=row['Name']  # Optional: Show name on click
    ).add_to(m)

# Save or display the map
m.save('porto_route_map.html')  # Save to HTML
m  # Display in Jupyter notebook (if applicable)
