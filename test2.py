import folium
import pandas as pd
import ast  # For safely evaluating the string as a list

# Load the data
df = pd.read_csv("outliers\outliers\outliers_20250523_110319.csv")

# Convert the POLYLINE string to actual coordinates
def parse_polyline(polyline_str):
    try:
        return ast.literal_eval(polyline_str)
    except:
        return []  # Return empty list if parsing fails

# Apply the parsing to all rows
df['coordinates'] = df['POLYLINE'].apply(parse_polyline)

# Create the base map centered on Porto
porto_coords = [41.1579, -8.6291]
m = folium.Map(location=porto_coords, zoom_start=14, tiles='OpenStreetMap')

# Add trajectories for all trips (or just the first few for visualization)
for idx, row in df.head(10000).iterrows():  # Just plotting first 10 trips for clarity
    if len(row['coordinates']) > 1:  # Only plot if there are valid coordinates
        # Convert coordinates to [lat, lon] format for Folium
        points = [[lat, lon] for [lon, lat] in row['coordinates']]
        
        # Add the trajectory to the map
        folium.PolyLine(
            locations=points,
            color='blue',
            weight=2,
            opacity=0.7,
            tooltip=f"Trip ID: {row['TRIP_ID']}"
        ).add_to(m)

        # Add markers for start and end points
        folium.CircleMarker(
            location=points[0],
            radius=5,
            color='green',
            fill=True,
            fill_color='green',
            fill_opacity=1,
            popup=f"Start of Trip {row['TRIP_ID']}"
        ).add_to(m)
        
        folium.CircleMarker(
            location=points[-1],
            radius=5,
            color='red',
            fill=True,
            fill_color='red',
            fill_opacity=1,
            popup=f"End of Trip {row['TRIP_ID']}"
        ).add_to(m)

# Save or display the map
m.save('porto_taxi_trips.html')
m  # Display in Jupyter notebook (if applicable)