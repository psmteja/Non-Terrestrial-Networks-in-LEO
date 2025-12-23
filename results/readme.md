# Temporal Graphs for LEO Satellites (SGP4 + Temporal APSP) — Summary for Professor

This document explains what the generated **graphs** and **logs** represent in our experiments that model a **LEO satellite constellation as a temporal graph** and compute **shortest communication-time paths** using the algorithm from the paper *“Computing All-Pairs Shortest Communication Time Paths for Periodically Varying Satellite Networks”*.

---

## 1) What the experiment is doing (high level)

We model a LEO constellation as a **temporal (time-varying) graph**:

- **Nodes (V)**: Satellites  
- **Edges (E_t)**: Inter-satellite links (ISLs) that exist **only when two satellites are within a chosen range threshold** at a given snapshot.

Time is discretized into **M snapshots**, each snapshot lasts **τ seconds**.  
So the observation window is:

\[
T = M \cdot \tau
\]

In our run logs:
- `tau = 60 seconds`
- `M = 32 snapshots`
- Total window: `T = 32 × 60 = 1920 seconds (32 minutes)`

For each snapshot we:
1. Propagate satellite positions using **SGP4** from TLEs
2. Compute pairwise distances
3. Build snapshot connectivity based on ISL range (e.g., 1000 km)
4. For small `n`, run the paper’s **temporal APSP shortest-time** algorithm

---

## 2) Meaning of the paper’s APSP output (5-node validation)

Log excerpt:

