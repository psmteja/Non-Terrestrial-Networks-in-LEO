# Non-Terrestrial-Networks-in-LEO


# Satellite Analysis Code Overview

This repository contains a Python script that analyzes satellite data using Two-Line Element (TLE) sets. The analysis is performed over time-buckets, focusing on connectivity, topology, and temporal properties of satellite networks.

## Analysis of Time-Buckets

For each time-bucket, the following analyses are performed:

### a. Count the Number of Distinct Satellites in Each Bucket
- The code iterates through each time bucket and counts the distinct satellites present. 
- It uses a set comprehension to identify unique satellite names, ensuring that duplicates are not counted multiple times. The result is printed for each bucket.

### b. Construct an Adjacency Matrix Corresponding to Each Bucket
- An adjacency matrix is constructed for each bucket to represent connectivity between satellites based on proximity.
- For each pair of satellites, their positions in Earth-Centered Inertial (ECI) coordinates are calculated using the `tle_to_eci` function.
- The code checks if the distance between satellites is within a defined link range (100 km):
  - If the distance is less than or equal to the link range, an edge is added between the satellites in the graph `G`, indicating they are "connected".

### c. Calculate Basic Topological Properties from the Adjacency Matrix of Each Bucket
- For each bucket, after constructing the adjacency matrix (as graph `G`):
  - **i. Node Degree**: The code calculates the degree of each node (satellite) and computes the maximum, minimum, and average degrees using the degree distribution.
  - **ii. Path Connectivity Measures**: The code calculates edge-disjoint and vertex-disjoint paths using NetworkX functions `edge_connectivity` and `node_connectivity`, determining the robustness of the network against node or edge failures.
  - **iii. Centrality Measures**: Various centrality metrics are computed:
    - **Betweenness Centrality**: Indicates a node's importance in connecting others.
    - **Closeness Centrality**: Measures how quickly a node can reach others in the network.
    - **Katz Centrality**: Accounts for the influence of nodes on each other, with a provision for convergence issues.
  - **iv. Connected Components**: The code identifies connected components within the graph and computes their sizes and diameters:
    - Max, min, and average sizes of components.
    - Max, min, and average diameters, providing insights into the network's structure.

### d. Extend Analyses to Temporal Versions Across Buckets
- The code prepares for future analyses of temporal properties across buckets, although specific implementations are not detailed in the provided script.

### e. Measure Temporal Properties of Nodes and Components
- The code tracks how long nodes remain neighbors across consecutive buckets. 
- For each satellite:
  - It initializes a structure to store neighbor counts and increments counts for each consecutive bucket where two satellites remain neighbors.
- After gathering this data, the code computes statistics for each satellite, including:
  - Total link duration with neighbors.
  - Maximum, minimum, and average link durations.
  - Identification of neighbors corresponding to maximum and minimum durations.

### f. Determine Churn Rates for Each Component
- The code tracks changes in components over time:
  - For each bucket, it constructs the graph and identifies connected components.
  - By comparing components from one bucket to the next, it calculates the "churn rate," indicating how many components changed or were reconfigured.

## Summary
This code systematically analyzes satellite data across time-buckets, focusing on connectivity, topology, and temporal properties. It constructs an adjacency matrix for each bucket, computes various graph metrics, and prepares for deeper temporal analyses to understand the dynamics of satellite networks over time.
