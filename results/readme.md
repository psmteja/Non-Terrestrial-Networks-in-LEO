# Temporal Graphs for LEO Satellites (SGP4 + Temporal APSP) — Summary + What the Logs/Graphs Mean

This README explains (1) what the code is doing, (2) what each **graph type** represents, and (3) how to interpret the **console logs** you shared, so you can forward it directly to your professor.

---

## 1) Goal of the experiment

The overall goal is to model a **LEO satellite constellation** as a **temporal graph** (time-varying network) and compute **shortest communication-time paths** using the algorithm described in the paper:

> *Computing All-Pairs Shortest Communication Time Paths for Periodically Varying Satellite Networks*

Key idea: satellite-to-satellite connectivity changes over time due to orbital motion, so the network must be treated as a **sequence of snapshots**, not a single static graph.

---

## 2) How the temporal graph is constructed

### 2.1 Nodes and time
- **Nodes (V):** satellites parsed from a TLE dataset
- **Time window:** discretized into `M` snapshots
- **Snapshot length:** `τ` seconds  
- Total window:  
  \[
  T = M \cdot \tau
  \]

From your logs:
- `τ = 60 seconds`
- `M = 32 snapshots`
- Total observed duration:
  - `T = 32 × 60 = 1920 seconds = 32 minutes`

### 2.2 Satellite positions (SGP4 propagation)
At each snapshot time `t_k`:
- For each satellite, we use **SGP4** propagation to compute ECI (TEME) position in km.
- This gives an array of positions:
  - `pos[k, i, :]` = 3D position of satellite `i` at snapshot `k`

### 2.3 Snapshot connectivity (edges)
For each snapshot `k`:
- Compute pairwise distances `D(i, j)` from the propagated positions.
- Define an ISL edge between satellites `i` and `j` if:
  - `distance(i, j) ≤ link_range_km`

This generates a per-snapshot adjacency:
- Snapshot graph \(G_k = (V, E_k)\)

### 2.4 Connectivity strings (paper model)
For each satellite pair `(p, q)`, we build a **connectivity string**:

- `CS_pq` is a binary vector of length `M`
- `CS_pq[k] = 1` if edge `(p, q)` exists during snapshot interval `k`, else `0`

This matches the paper’s temporal adjacency definition:
- \(A(p,q) = (CS_{pq}, d_{pq}, t_{pq})\)

---

## 3) Paper algorithm: “Shortest time” APSP on temporal graphs

### 3.1 What “shortest time path” means
A shortest time (fastest) path is the one with **minimum total time spent from departure to arrival**, which includes:
- transmission/traversal time across edges
- possible **waiting** at intermediate nodes until a valid next edge exists

This differs from:
- shortest hop-count path
- shortest distance path
- earliest arrival path

### 3.2 What `start_I`, `finish_I`, and `pred` mean
The APSP output stores for each pair `(src, dst)`:

- `start_I`: snapshot interval index when the message starts at `src`
- `finish_I`: snapshot interval index when it arrives at `dst`
- `duration_intervals = finish_I − start_I + 1`
- `pred`: predecessor of `dst` on the chosen optimal temporal path
- `path_nodes`: reconstructed route using predecessor pointers

The algorithm internally generates many feasible contact intervals; then it selects the one with **minimum duration** as the shortest-time path.

### 3.3 Why APSP becomes expensive
Paper time complexity:
\[
O(M^2 n^3)
\]
So for large `n`, APSP becomes computationally expensive on CPU.

---

## 4) What the console logs mean (your output)

Below is the interpretation of each section of your logs.

---


PAPER 5-NODE EXAMPLE (APSP)

1->2: (start=3, finish=4, pred=1) path=1->2  
1->3: (start=3, finish=6, pred=2) path=1->2->3  
...

**Meaning:**  
This section validates the implementation against the paper’s small example temporal graph.

Example line:
- `1->3: (start=3, finish=6, pred=2) path=1->2->3`

Interpretation:
- The best (fastest) temporal path from node 1 to node 3 starts at snapshot 3
- Arrives at snapshot 6
- Total duration = `6−3+1 = 4` snapshot intervals
- Final step arrives at node 3 from node 2 (`pred=2`)

This demonstrates temporal routing where connectivity may appear/disappear over time.

---

### 4.2 N=10 satellites run

Loaded N=10 satellites, propagated M=32 snapshots (tau=60.0s).  
Saved: plots\sat_n10_snapshot_t000.png  
Saved: plots_cartopy\sat_n10_map_t000.png  
...  
Saved: apsp_shortest_time_paths_n10.csv  

Top-10 shortest-time paths (by duration):

| src_name   | dst_name   | start_I | finish_I | duration_intervals | path_nodes |
|-----------|------------|--------:|---------:|-------------------:|-----------|
| COURIER 1B | TRANSIT 4A | 10.0 | 11.0 | 2.0 | 6->8 |
| TRANSIT 4A | COURIER 1B | 10.0 | 11.0 | 2.0 | 8->6 |

**Meaning:**
- The code processed 10 satellites for 32 snapshots (32 minutes)
- It saved two types of plots repeatedly at different times (`t000, t004, ..., t028`)
- It computed temporal APSP results and wrote them to:
  - `apsp_shortest_time_paths_n10.csv`

The “Top-10 shortest-time paths” shows the fastest pairs:
- Example:
  - `duration_intervals = 2` means 2 snapshots
  - With `τ=60s`, duration ≈ `2 × 60 = 120 seconds`
  - Path is direct from index 6 to index 8 (`6->8`)

---

### 4.3 N=50 satellites run

Loaded N=50 satellites, propagated M=32 snapshots (tau=60.0s).  
Saved: plots\sat_n50_snapshot_t000.png  
Saved: plots_cartopy\sat_n50_map_t000.png  
...  
Saved: apsp_shortest_time_paths_n50.csv  

Top-10 shortest-time paths (by duration):
- `duration_intervals = 2.0`, `path_nodes = 2->21`
- `duration_intervals = 2.0`, `path_nodes = 2->25`
...

**Meaning:**
- Same workflow, now for 50 satellites.
- APSP was computed (still feasible on CPU at this size).
- The Top-10 table again highlights fastest communication-time paths.

---

### 4.4 N=500 satellites run (plots only, APSP skipped)

Loaded N=500 satellites, propagated M=32 snapshots (tau=60.0s).  
Saved: plots\sat_n500_snapshot_t000.png  
Saved: plots_cartopy\sat_n500_map_t000.png  
...  
Skipping APSP for N=500 (too expensive on CPU: O(M^2*n^3)).  

**Meaning:**
- The constellation was scaled to 500 satellites (subset of your larger dataset)
- The system generated snapshots + plots successfully
- APSP was skipped because the paper’s algorithm is too expensive at `n=500` on CPU:
  \[
  O(M^2 n^3)
  \]

This also matches the paper’s motivation: the algorithm is structured for GPU parallelization.

---

## 5) What the generated graphs are (and what each means)

You saved **two** categories of figures for each N and each snapshot time.

---

### 5.1 “Snapshot Graph” figures (Network topology view)

Files like:
- `plots/sat_n10_snapshot_t008.png`
- `plots/sat_n50_snapshot_t008.png`
- `plots/sat_n500_snapshot_t008.png`

**What they show:**
- A **graph view** of the snapshot topology at time index `t`
- Nodes are satellites (labels often show satellite names)
- Edges are ISLs present at that snapshot based on range threshold
- Node positions come from a graph layout algorithm (e.g., spring layout), not geography

**Why it is useful:**
- Shows connectivity structure: clusters, components, isolated satellites
- Helps analyze network properties (connectivity, degree patterns, components)

**Interpretation sentence (paper-ready):**  
“Figure X shows the snapshot connectivity graph \(G_t\) at time index \(t\), where edges represent inter-satellite links available under the ISL range constraint.”

---

### 5.2 “World Map” figures (Geospatial / physical view)

Files like:
- `plots_cartopy/sat_n10_map_t008.png`
- `plots_cartopy/sat_n50_map_t008.png`
- `plots_cartopy/sat_n500_map_t008.png`

**What they show:**
- A geographic world map using Cartopy
- Satellite locations are plotted by their projected latitude/longitude at snapshot `t`
- Optional edges show which satellites are connected (within range) at that same instant

**Why it is useful:**
- Shows how connectivity changes *as satellites move over Earth*
- Makes results intuitive and “communication-network on orbit” style

**Interpretation sentence (paper-ready):**  
“Figure Y overlays satellite positions on a global map at snapshot \(t\), highlighting spatial distribution and active inter-satellite links under the distance threshold.”

---

## 6) Why there are multiple times per run (t000, t004, t008, ...)

Your log shows plots saved at:
- `t000, t004, t008, t012, ..., t028`

This means you sampled every 4th snapshot (or a chosen step size) for visualization.  
These sequential frames demonstrate **temporal evolution** of the network.

---

## 7) Final paper-ready one-sentence summary

“We modeled a LEO satellite constellation as a temporal graph by propagating satellite positions using SGP4 and constructing per-snapshot ISL connectivity from a distance threshold, validated the temporal APSP shortest-time algorithm on the paper’s 5-node example, computed APSP for small constellations (N=10, N=50), and visualized both graph-topology snapshots and geospatial Cartopy maps to illustrate time-varying connectivity and routing behavior.”

---


