import numpy as np
import pandas as pd
import networkx as nx
from datetime import datetime, timedelta
from math import radians, sin, cos, sqrt, atan2

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

# Function to convert TLE epoch to datetime
def epoch_to_datetime(epoch):
    year = 2000 + int(epoch // 1000)  # Adjust based on TLE years
    day_of_year = int(epoch % 1000)
    dt = datetime(year, 1, 1) + timedelta(days=day_of_year - 1)  # Adjust to start from 0
    fractional_day = epoch % 1
    dt += timedelta(seconds=fractional_day * 86400)  # 86400 seconds in a day
    return dt

# Function to convert orbital elements to ECI coordinates
def tle_to_eci(satellite, time):
    # Constants
    mu = 398600  # Earth's gravitational parameter, km^3/s^2
    r_earth = 6371  # Earth's radius in km

    # Orbital parameters
    a = (mu / (satellite['mean_motion'] * 2 * np.pi / 86400) ** 2) ** (1/3)  # Semi-major axis
    e = satellite['eccentricity']
    i = radians(satellite['inclination'])
    raan = radians(satellite['raan'])
    arg_perigee = radians(satellite['arg_of_perigee'])
    M0 = radians(satellite['mean_anomaly'])

    # Mean anomaly at the given time
    n = satellite['mean_motion'] * (2 * np.pi / 86400)  # Mean motion in radians per second
    M = M0 + n * (time - epoch_to_datetime(float(satellite['epoch'])).timestamp())

    # Solve Kepler's equation for eccentric anomaly
    E = M  # Initial guess
    for _ in range(10):  # Iterate to solve for E
        E = M + e * sin(E)

    # Calculate position in the orbital plane
    x = a * (cos(E) - e)
    y = a * sqrt(1 - e**2) * sin(E)

    # Transform to ECI coordinates
    r = sqrt(x**2 + y**2)
    theta = atan2(y, x)

    # ECI coordinates
    pos_eci = [
        r * (cos(raan) * cos(theta + arg_perigee) - sin(raan) * sin(theta + arg_perigee) * cos(i)),
        r * (sin(raan) * cos(theta + arg_perigee) + cos(raan) * sin(theta + arg_perigee) * cos(i)),
        r * (sin(theta + arg_perigee) * sin(i))
    ]
    return pos_eci

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

    # b. Construct adjacency matrix based on distance
    G = nx.Graph()
    for s1 in satellites:
        pos1 = tle_to_eci(s1, time_buckets[bucket_index].timestamp())
        G.add_node(s1['name'], pos=pos1)
        for s2 in satellites:
            if s1['name'] != s2['name']:
                pos2 = tle_to_eci(s2, time_buckets[bucket_index].timestamp())
                distance = sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2 + (pos1[2] - pos2[2]) ** 2)
                # Assuming link range if distance is less than a threshold (e.g., 100 km)
                if distance <= 100:
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
            try:
                katz_centrality = nx.katz_centrality(G, alpha=0.1)
            except nx.PowerIterationFailedConvergence:
                print("Katz centrality calculation failed to converge.")
                katz_centrality = {}
        else:
            # Use largest connected component for centrality measures
            largest_component = max(nx.connected_components(G), key=len)
            G_largest = G.subgraph(largest_component)
            betweenness_centrality = nx.betweenness_centrality(G_largest)
            closeness_centrality = nx.closeness_centrality(G_largest)
            try:
                katz_centrality = nx.katz_centrality(G_largest, alpha=0.1)
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
                # Initialize the neighbor count if it doesn't exist
                if s2['name'] not in link_duration_stats[s1['name']]:
                    link_duration_stats[s1['name']][s2['name']] = 0

                # Check if s2 was also a neighbor in the previous bucket
                if bucket_index > 0 and s2['name'] in link_duration_stats:
                    if s1['name'] in link_duration_stats[s2['name']]:
                        link_duration_stats[s1['name']][s2['name']] += 1
                    # No else needed here, as we already initialized above
                else:
                    link_duration_stats[s1['name']][s2['name']] += 1  # Increment based on current bucket

# Calculate average, max, and min link duration for each satellite
link_duration_summary = {}
for satellite, neighbors in link_duration_stats.items():
    total_duration = sum(neighbors.values())

    # Check if neighbors are present to avoid ValueError
    if neighbors:
        max_duration = max(neighbors.values())
        min_duration = min(neighbors.values())
        avg_duration = total_duration / len(neighbors) if neighbors else 0
    else:
        max_duration = min_duration = avg_duration = 0  # Default values if no neighbors

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
                pos1 = tle_to_eci(s1, time_buckets[bucket_index].timestamp())
                pos2 = tle_to_eci(s2, time_buckets[bucket_index].timestamp())
                distance = sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2 + (pos1[2] - pos2[2]) ** 2)
                if distance <= 100:  # Link range threshold
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


import matplotlib.pyplot as plt

# Function to visualize distinct satellites
def plot_distinct_satellites(bucket_indices, distinct_counts):
    plt.figure(figsize=(12, 6))
    plt.bar(bucket_indices, distinct_counts, color='blue')
    plt.title('Distinct Satellites per Bucket')
    plt.xlabel('Bucket Index')
    plt.ylabel('Number of Distinct Satellites')
    plt.xticks(bucket_indices)
    plt.grid()
    plt.show()

# Function to visualize the degree distribution
def plot_degree_distribution(bucket_indices, max_degrees, avg_degrees):
    plt.figure(figsize=(12, 6))
    plt.plot(bucket_indices, max_degrees, label='Max Degree', marker='o')
    plt.plot(bucket_indices, avg_degrees, label='Avg Degree', marker='o')
    plt.title('Degree Distribution per Bucket')
    plt.xlabel('Bucket Index')
    plt.ylabel('Degree')
    plt.xticks(bucket_indices)
    plt.legend()
    plt.grid()
    plt.show()

# Prepare data for visualization
distinct_counts = []
max_degrees = []
avg_degrees = []

# Analyze each time-bucket and collect data for visualization
for bucket_index, satellites in bucketed_satellites.items():
    # Count distinct satellites
    distinct_satellites = len(set(s['name'] for s in satellites))
    distinct_counts.append(distinct_satellites)

    # Construct adjacency matrix based on distance
    G = nx.Graph()
    for s1 in satellites:
        pos1 = tle_to_eci(s1, time_buckets[bucket_index].timestamp())
        G.add_node(s1['name'], pos=pos1)
        for s2 in satellites:
            if s1['name'] != s2['name']:
                pos2 = tle_to_eci(s2, time_buckets[bucket_index].timestamp())
                distance = sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2 + (pos1[2] - pos2[2]) ** 2)
                if distance <= 100:
                    G.add_edge(s1['name'], s2['name'])

    # Calculate topological properties
    if G.number_of_nodes() > 0:
        degrees = dict(G.degree())
        max_degree = max(degrees.values(), default=0)
        avg_degree = np.mean(list(degrees.values())) if degrees else 0

        max_degrees.append(max_degree)
        avg_degrees.append(avg_degree)

# Plot distinct satellites
bucket_indices = list(range(num_buckets))
plot_distinct_satellites(bucket_indices, distinct_counts)

# Plot degree distribution
plot_degree_distribution(bucket_indices, max_degrees, avg_degrees)

# Function to visualize the graph for a specific bucket
def visualize_graph(G, bucket_index):
    plt.figure(figsize=(12, 12))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=500, font_size=10)
    plt.title(f"Satellite Graph for Bucket {bucket_index}")
    plt.show()

# Visualize the graph for the first bucket as an example
if bucketed_satellites[0]:  # Ensure the first bucket is not empty
    G_first = nx.Graph()
    for s1 in bucketed_satellites[0]:
        pos1 = tle_to_eci(s1, time_buckets[0].timestamp())
        G_first.add_node(s1['name'], pos=pos1)
        for s2 in bucketed_satellites[0]:
            if s1['name'] != s2['name']:
                pos2 = tle_to_eci(s2, time_buckets[0].timestamp())
                distance = sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2 + (pos1[2] - pos2[2]) ** 2)
                if distance <= 100:
                    G_first.add_edge(s1['name'], s2['name'])
    visualize_graph(G_first, 0)  # Visualize for the first bucket
