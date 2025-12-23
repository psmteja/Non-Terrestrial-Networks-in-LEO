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

### 4.1 Paper toy example verification (5 nodes)

