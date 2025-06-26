import numpy as np
import pandas as pd
import networkx as nx
from datetime import datetime, timedelta


# Function to load TLE data from a text file
def load_tle_data(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()

    tle_data = []
    for i in range(0, len(lines), 3):
        if i + 2 < len(lines):
            name_line = lines[i].strip()
            first_line = lines[i + 1].strip()
            second_line = lines[i + 2].strip()
            satellite = {
                'name': name_line.split()[1],  # Extract satellite name
                'epoch': first_line[18:32].strip(),  # Extract epoch
                'inclination': float(second_line[8:16].strip()),  # Extract inclination
                'raan': float(second_line[17:25].strip()),  # Extract RAAN
                'eccentricity': float(second_line[26:33].strip()) / 1e7,  # Extract eccentricity
                'arg_of_perigee': float(second_line[34:42].strip()),  # Extract argument of perigee
                'mean_anomaly': float(second_line[43:51].strip()),  # Extract mean anomaly
                'mean_motion': float(second_line[52:63].strip())  # Extract mean motion
            }
            tle_data.append(satellite)
    return tle_data


# Load TLE data from the specified text file
filename = 'leo_data.txt'  # Replace with your actual file path
tle_data = load_tle_data(filename)

# Check if TLE data is loaded correctly
print(f"Loaded {len(tle_data)} satellites")

# Create time-buckets
num_buckets = 144
bucket_duration = timedelta(minutes=10)
start_time = datetime(2025, 6, 23)  # Adjust start time for buckets to match your data
time_buckets = [start_time + i * bucket_duration for i in range(num_buckets)]

# Function to convert TLE epoch to datetime
def epoch_to_datetime(epoch):
    year = 2000 + int(epoch // 1000)  # Adjust based on TLE years
    day_of_year = int(epoch % 1000)
    dt = datetime(year, 1, 1) + timedelta(days=day_of_year - 1)  # Adjust to start from 0
    fractional_day = epoch % 1
    dt += timedelta(seconds=fractional_day * 86400)  # 86400 seconds in a day
    return dt


# Organize satellites into buckets based on their epochs
bucketed_satellites = {i: [] for i in range(num_buckets)}
for satellite in tle_data:
    epoch_datetime = epoch_to_datetime(float(satellite['epoch']))
    for i, bucket_time in enumerate(time_buckets):
        if bucket_time <= epoch_datetime < bucket_time + bucket_duration:
            bucketed_satellites[i].append(satellite)
            break

# Analyze each time-bucket
for bucket_index, satellites in bucketed_satellites.items():
    # a. Count distinct satellites
    distinct_satellites = len(set(s['name'] for s in satellites))
    print(f"Bucket {bucket_index}: Distinct Satellites = {distinct_satellites}")

    # b. Construct adjacency matrix
    G = nx.Graph()
    for s1 in satellites:
        G.add_node(s1['name'])
        for s2 in satellites:
            if s1['name'] != s2['name']:
                # Assuming link if the inclination difference is within a threshold
                if abs(s1['inclination'] - s2['inclination']) < 1:  # Example threshold
                    G.add_edge(s1['name'], s2['name'])

    # c. Calculate topological properties
    if G.number_of_nodes() > 0:  # Check if the graph is not empty
        degrees = dict(G.degree())
        max_degree = max(degrees.values(), default=0)
        min_degree = min(degrees.values(), default=0)
        avg_degree = np.mean(list(degrees.values())) if degrees else 0

        # i. Path connectivity measures
        edge_disjoint_paths = nx.edge_connectivity(G)
        vertex_disjoint_paths = nx.node_connectivity(G)

        # ii. Centrality measures
        if nx.is_connected(G):
            betweenness_centrality = nx.betweenness_centrality(G)
            closeness_centrality = nx.closeness_centrality(G)
            katz_centrality = nx.katz_centrality(G, alpha=0.1)
        else:
            # Use largest connected component for centrality measures
            largest_component = max(nx.connected_components(G), key=len)
            G_largest = G.subgraph(largest_component)
            betweenness_centrality = nx.betweenness_centrality(G_largest)
            closeness_centrality = nx.closeness_centrality(G_largest)
            try:
                katz_centrality = nx.katz_centrality(G_largest, alpha=0.05, max_iter=5000)
            except nx.PowerIterationFailedConvergence:
                print("Katz centrality calculation failed to converge.")
                katz_centrality = {}

        # iii. Connected components
        connected_components = list(nx.connected_components(G))
        component_sizes = [len(comp) for comp in connected_components]
        sizes_max = max(component_sizes, default=0)
        sizes_min = min(component_sizes, default=0)
        sizes_avg = np.mean(component_sizes) if component_sizes else 0

        # Component diameter
        diameters = [nx.diameter(G.subgraph(comp)) for comp in connected_components if len(comp) > 1]
        diameter_max = max(diameters, default=0)
        diameter_min = min(diameters, default=0)
        diameter_avg = np.mean(diameters) if diameters else 0
    else:
        max_degree = min_degree = avg_degree = 0
        edge_disjoint_paths = vertex_disjoint_paths = 0
        sizes_max = sizes_min = sizes_avg = 0
        diameter_max = diameter_min = diameter_avg = 0

    # Print calculated properties
    print(f"Bucket {bucket_index}: Max Degree = {max_degree}, Min Degree = {min_degree}, Avg Degree = {avg_degree}")
    print(f"Bucket {bucket_index}: Max Component Size = {sizes_max}, Min Component Size = {sizes_min}, Avg Component Size = {sizes_avg}")
    print(f"Bucket {bucket_index}: Max Diameter = {diameter_max}, Min Diameter = {diameter_min}, Avg Diameter = {diameter_avg}")
    print(f"Bucket {bucket_index}: Edge-Disjoint Paths = {edge_disjoint_paths}, Vertex-Disjoint Paths = {vertex_disjoint_paths}")

# e. Measure temporal properties of nodes and components
link_duration_stats = {}
for bucket_index in range(num_buckets):
    satellites = bucketed_satellites[bucket_index]
    for s1 in satellites:
        if s1['name'] not in link_duration_stats:
            link_duration_stats[s1['name']] = {}

        for s2 in satellites:
            if s1['name'] != s2['name']:
                # Check if s2 was also a neighbor in the previous bucket
                if bucket_index > 0 and s2['name'] in link_duration_stats:
                    if s1['name'] in link_duration_stats[s2['name']]:
                        link_duration_stats[s1['name']][s2['name']] += 1
                    else:
                        link_duration_stats[s1['name']][s2['name']] = 1
                else:
                    link_duration_stats[s1['name']][s2['name']] = 1

# Calculate average, max, and min link duration for each satellite
link_duration_summary = {}
for satellite, neighbors in link_duration_stats.items():
    total_duration = sum(neighbors.values())
    max_duration = max(neighbors.values())
    min_duration = min(neighbors.values())
    avg_duration = total_duration / len(neighbors) if neighbors else 0

    link_duration_summary[satellite] = {
        'total_duration': total_duration,
        'max_duration': max_duration,
        'min_duration': min_duration,
        'avg_duration': avg_duration,
        'neighbors': neighbors
    }

# Print link duration statistics for each satellite
for satellite, stats in link_duration_summary.items():
    print(f"Satellite: {satellite}, Total Duration: {stats['total_duration']}, "
          f"Max Duration: {stats['max_duration']}, Min Duration: {stats['min_duration']}, "
          f"Avg Duration: {stats['avg_duration']}, Neighbors: {list(stats['neighbors'].keys())}")

# f. Calculate churn rates for each connected component
component_churn_rates = {}
component_states = {}

# Track components over time
for bucket_index in range(num_buckets):
    satellites = bucketed_satellites[bucket_index]
    G = nx.Graph()
    for s1 in satellites:
        G.add_node(s1['name'])
        for s2 in satellites:
            if s1['name'] != s2['name']:
                if abs(s1['inclination'] - s2['inclination']) < 1:  # Example threshold for adjacency
                    G.add_edge(s1['name'], s2['name'])

    # Determine the connected components for this bucket
    components = list(nx.connected_components(G))
    component_states[bucket_index] = [frozenset(comp) for comp in components]

# Calculate churn by comparing component states over consecutive buckets
for bucket_index in range(1, num_buckets):
    current_components = set(component_states[bucket_index])
    previous_components = set(component_states[bucket_index - 1])

    # Calculate churn by comparing component sets
    churn = len(current_components.symmetric_difference(previous_components))
    component_churn_rates[bucket_index] = churn

# Print churn rates for each bucket
for bucket_index, churn in component_churn_rates.items():
    print(f"Bucket {bucket_index}: Churn Rate = {churn}")
