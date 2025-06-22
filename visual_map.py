import folium
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

# Sample latitude and longitude points for Porto (you can replace with your data)
data = {
    'Name': ['Porto Cathedral', 'Clérigos Tower', 'Livraria Lello', 'Ribeira Square', 'Casa da Música'],
    'Latitude': [41.1425, 41.1458, 41.1469, 41.1408, 41.1589],
    'Longitude': [-8.6114, -8.6150, -8.6153, -8.6119, -8.6308]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Create a GeoDataFrame
geometry = [Point(xy) for xy in zip(df['Longitude'], df['Latitude'])]
gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")

# Create a base map centered on Porto
porto_coords = [41.1579, -8.6291]  # Approximate center of Porto
m = folium.Map(location=porto_coords, zoom_start=14, tiles='OpenStreetMap')

# Add points to the map
for idx, row in gdf.iterrows():
    folium.Marker(
        location=[row['Latitude'], row['Longitude']],
        popup=row['Name'],
        icon=folium.Icon(color='blue', icon='info-sign')
    ).add_to(m)

# Add a polygon for Porto's boundary (optional - requires shapefile)
# This is just an example - you would need the actual boundary data
try:
    # Example of how you might add city boundaries if you had the data
    # porto_boundary = gpd.read_file('path_to_porto_shapefile.shp')
    # folium.GeoJson(porto_boundary).add_to(m)
    pass
except:
    print("Could not load boundary data - proceeding without it")

# Display the map
m.save('porto_map.html')  # Save to HTML file
m  # Display in notebook if using Jupyter