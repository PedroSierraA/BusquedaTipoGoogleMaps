import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, LineString
import networkx as nx
import osmnx as ox
import geopy
from geopy.geocoders import Nominatim
import matplotlib.pyplot as plt

# This dictionary defines the weights for different road surface types.
surface_weights = {
    'asphalt': 1.0,
    'concrete': 1.1,
    'concrete:lanes': 1.1,
    'concrete:plates': 1.2,
    'paved': 1.2,
    'paving_stones': 1.4,
    'fine_gravel': 1.6,
    'gravel': 2.0,
    'cobblestone': 2.2,
    'ground': 2.5,
    'dirt': 3.0,
    'unpaved': 2.8,
    'unknown': 4.0
}

def add_surface_data_safe(gdf_edges, gdf_surface):
    """
    Safely adds surface data to the graph edges.
    Assigns a default surface type and then updates it with actual data where available.
    """
    edges_with_surface = gdf_edges.copy()
    # Initialize with a default surface type
    edges_with_surface['surface'] = 'asphalt'

    # Update with actual surface data where intersections are found
    for idx, surface_feature in gdf_surface.iterrows():
        if pd.isna(surface_feature['surface']):
            continue
        mask = edges_with_surface.geometry.intersects(surface_feature.geometry)
        edges_with_surface.loc[mask, 'surface'] = surface_feature['surface']

    return edges_with_surface

def add_proximity_features(G, gdf_features, feature_name, buffer_distance=0.001):
    """
    Adds a count of nearby features (like hospitals, schools) to each edge in the graph.
    """
    for u, v, key, data in G.edges(keys=True, data=True):
        edge_geom = data['geometry']
        buffer_zone = edge_geom.buffer(buffer_distance)

        # Count features within the buffer zone
        nearby_features = gdf_features[gdf_features.geometry.intersects(buffer_zone)]
        count = len(nearby_features)

        # Add the count to the edge data
        G.edges[u, v, key][feature_name] = count

# --- Heuristic Weight Functions ---

def travel_time_weight(u, v, data):
    """
    Heuristic based on the edge's travel time.
    """
    return data.get('travel_time', 1)

def surface_weight(u, v, data):
    """
    Heuristic that calculates weight based on road surface type and length.
    Prefers smoother surfaces.
    """
    surface = data.get('surface', 'unknown')
    length = data.get('length', 1)
    factor = surface_weights.get(surface, 4.0)

    weight = length * factor
    return weight

def force_hospital_weight(u, v, data):
    """
    Heuristic that heavily penalizes routes that do not pass near a hospital.
    """
    length = data.get('length', 1)
    hospitals = data.get('nearby_hospitals', 0)
    surface = data.get('surface', 'unknown')

    surface_factor = surface_weights.get(surface, 4.0)

    # Apply a very large penalty if no hospital is nearby
    hospital_penalty = 1000 if hospitals == 0 else 1.0

    weight = length * surface_factor * hospital_penalty
    return weight

def create_graph(place_name="Envigado, Antioquia, Colombia"):
    """
    Creates and enriches the road network graph for the specified place.
    """
    print(f"Creating graph for {place_name}...")
    # Download the graph data from OpenStreetMap
    G = ox.graph_from_place(place_name, network_type='drive', simplify=False)

    # Add edge speeds and travel times
    G = ox.add_edge_speeds(G)
    G = ox.add_edge_travel_times(G)

    # Convert graph to GeoDataFrames to work with spatial data
    gdf_nodes, gdf_edges = ox.graph_to_gdfs(G)

    # Add road surface information
    print("Adding surface data...")
    gdf_surface = ox.features.features_from_place(place_name, tags={'surface': True})
    gdf_edges_with_surface = add_surface_data_safe(gdf_edges, gdf_surface)

    # Recreate the graph with the new surface data
    G_surface = ox.graph_from_gdfs(gdf_nodes, gdf_edges_with_surface)

    # Add proximity features for amenities
    print("Adding proximity features (hospitals, schools, parks)...")
    gdf_hospitals = ox.features.features_from_place(place_name, tags={'amenity': 'hospital'})
    add_proximity_features(G_surface, gdf_hospitals, 'nearby_hospitals')

    gdf_schools = ox.features.features_from_place(place_name, tags={'amenity': 'school'})
    add_proximity_features(G_surface, gdf_schools, 'nearby_schools')

    gdf_parks = ox.features.features_from_place(place_name, tags={'leisure': 'park'})
    add_proximity_features(G_surface, gdf_parks, 'crosses_park')

    print("Graph creation complete.")
    return G_surface

def find_route(start_location, end_location, heuristic='travel_time'):
    """
    Finds the best route between two locations based on a selected heuristic.

    Args:
        start_location (str): The starting address.
        end_location (str): The destination address.
        heuristic (str): The heuristic to use for routing.
                         Options: 'travel_time', 'surface', 'hospital'.

    Returns:
        tuple: A tuple containing the graph and the route (list of node IDs).
               Returns (None, None) if a route cannot be found.
    """
    # Create the graph
    G = create_graph()

    # Set up the geocoder
    locator = Nominatim(user_agent='myGeocoder')

    # Geocode start and end locations
    print(f"Geocoding '{start_location}' and '{end_location}'...")
    start_point = locator.geocode(start_location)
    end_point = locator.geocode(end_location)

    if not start_point or not end_point:
        print("Error: One or both locations could not be geocoded. Please try again.")
        return None, None

    start_coords = (start_point.latitude, start_point.longitude)
    end_coords = (end_point.latitude, end_point.longitude)

    # Find the nearest nodes in the graph to the geocoded points
    start_node = ox.distance.nearest_nodes(G, start_coords[1], start_coords[0])
    end_node = ox.distance.nearest_nodes(G, end_coords[1], end_coords[0])

    # Define the available heuristics
    heuristics = {
        "travel_time": travel_time_weight,
        "surface": surface_weight,
        "hospital": force_hospital_weight
    }

    weight_function = heuristics.get(heuristic)
    if not weight_function:
        print(f"Error: Invalid heuristic '{heuristic}'.")
        print(f"Available options are: {list(heuristics.keys())}")
        return G, None

    # Calculate the shortest path using the selected heuristic
    print(f"Calculating route using '{heuristic}' heuristic...")
    try:
        route = nx.shortest_path(G, start_node, end_node, weight=weight_function)
        print("Route found successfully.")
        return G, route
    except nx.NetworkXNoPath:
        print("Error: No path could be found between the specified locations.")
        return G, None

if __name__ == '__main__':
    print("--- Waze/Google Maps Style Route Finder ---")

    start_location = input("Enter the starting location (e.g., 'Sede Posgrados Eia, Envigado, Colombia'): ")
    end_location = input("Enter the ending location (e.g., 'Universidad EIA, Envigado, Colombia'): ")
    heuristic = input("Choose a heuristic (travel_time, surface, hospital): ")

    # Use default values if input is empty
    if not start_location:
        start_location = 'Sede Posgrados Eia, Envigado, Colombia'
    if not end_location:
        end_location = 'Universidad EIA, Envigado, Colombia'
    if not heuristic:
        heuristic = 'travel_time'

    # Find the route
    graph, route = find_route(start_location, end_location, heuristic)

    if route:
        print(f"\nRoute found with {len(route)} nodes.")
        print("Plotting route... (Close the plot window to exit)")
        # Plot the route
        fig, ax = ox.plot_graph_route(graph, route, route_linewidth=6, node_size=0, bgcolor='k')
        plt.show()
    else:
        print("\nCould not find a route. Please check the locations and try again.")
