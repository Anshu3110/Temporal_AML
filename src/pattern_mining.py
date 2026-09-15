"""Graph topology pattern mining for multi-typology AML detection.

Mines three canonical money laundering typologies from transaction graph topology:
1. Circular Transfers (y_circ): Cyclic transaction loops (directed cycles).
2. Layering (y_lay): Sequential peeling chains across time steps.
3. Smurfing (y_smurf): Fan-in/fan-out structuring with high degree concentration.

Generates aligned [N, 3] multi-label boolean targets for multi-task training.
"""

from collections import defaultdict
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import networkx as nx
import numpy as np
import torch
import sys

# Ensure workspace root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src.data_loader import EllipticDatasetLoader
except ModuleNotFoundError:
    from data_loader import EllipticDatasetLoader


# ==============================================================================
# 1. Circular Transfer Detection
# ==============================================================================
def find_circular_labels(
    G: nx.DiGraph,
    max_cycle_len: int = 5,
    ego_radius: int = 3,
) -> Dict[int, bool]:
    """Identifies nodes participating in directed cycles of length <= max_cycle_len.

    Because enumerating all simple cycles across a 200,000+ node graph is intractable
    (exponential worst-case complexity), this function applies an exact decomposition:
    1. A node participates in a directed cycle of length >= 2 if and only if it belongs
       to a Strongly Connected Component (SCC) of size >= 2. All SCCs of size 1 (without
       self-loops) are guaranteed to be acyclic and are pruned in O(V + E) time.
    2. For nodes inside SCCs of size >= 2, cycle search is restricted to a bounded
       ego-network (radius <= ego_radius, bounded DFS up to max_cycle_len), or
       `nx.simple_cycles` on small component subgraphs.
    3. Self-loops (length 1 <= max_cycle_len) are directly identified.

    Note on the Elliptic Bitcoin dataset:
    Raw Bitcoin transactions form a Directed Acyclic Graph (DAG) by UTXO design,
    meaning raw transaction-level directed cycles are nonexistent unless addresses
    or reciprocal settlement flows are mapped. This function strictly implements
    the cycle detector and flags positive nodes whenever cyclic topology exists.

    Args:
        G: Directed transaction graph (nx.DiGraph) with integer node keys 0..N-1.
        max_cycle_len: Maximum length of directed cycle considered circular laundering.
        ego_radius: Maximum BFS search radius for local cycle extraction.

    Returns:
        Dict[int, bool]: Mapping from node_id to boolean indicator (True = participating in cycle).
    """
    labels: Dict[int, bool] = {n: False for n in G.nodes()}

    # 1. Check self-loops (length 1)
    for n in nx.nodes_with_selfloops(G):
        labels[n] = True

    # 2. Decompose into Strongly Connected Components (Tarjan / Kosaraju in O(V + E))
    sccs = [c for c in nx.strongly_connected_components(G) if len(c) >= 2]

    if not sccs:
        return labels

    # 3. For non-trivial SCCs, perform bounded search up to max_cycle_len
    for scc in sccs:
        scc_subgraph = G.subgraph(scc)

        if len(scc) <= 100:
            # Exact simple cycles on small components
            try:
                for cycle in nx.simple_cycles(scc_subgraph):
                    if len(cycle) <= max_cycle_len:
                        for u in cycle:
                            labels[u] = True
            except Exception:
                pass
        else:
            # Bounded DFS from each node in the component
            for root in scc:
                if labels[root]:
                    continue
                # Local DFS up to depth max_cycle_len
                stack = [(root, [root])]
                visited_in_path = {root}
                while stack:
                    curr, path = stack.pop()
                    if len(path) > max_cycle_len:
                        continue
                    for nbr in scc_subgraph.successors(curr):
                        if nbr == root and len(path) >= 2:
                            for node in path:
                                labels[node] = True
                            break
                        if nbr not in visited_in_path and len(path) < max_cycle_len:
                            visited_in_path.add(nbr)
                            stack.append((nbr, path + [nbr]))

    return labels


# ==============================================================================
# 2. Layering (Peeling Chain) Detection
# ==============================================================================
def find_layering_labels(
    G: nx.DiGraph,
    min_path_len: int = 4,
    max_fanout: int = 3,
    intermediary_only: bool = True,
    strict_time: bool = False,
) -> Dict[int, bool]:
    """Identifies nodes that lie on a directed peeling chain of length >= min_path_len.

    Forensic & Graph-Mining Rationale (Literature-Derived):
    Peeling chains are the quintessential Bitcoin money laundering obfuscation typology
    (Meiklejohn et al., IMC 2013; Möser et al., 2013; Ron & Shamir, 2013; Wu et al., IEEE TIFS 2021).
    A peeling chain exhibits three foundational structural constraints:
    1. Constrained Fan-Out (max_fanout <= 3): At each hop, a peeling transaction splits funds
       into a small peeled payment and a majority change output forwarded to a fresh address.
       Transactions with high fan-out (> 3) represent commercial exchange batch distributions,
       not laundering peeling sequences. We constrain edges to transitions where source out-degree <= max_fanout.
    2. Sequential Depth (min_path_len >= 4): Obfuscation sequences require multiple consecutive
       hops (typically >= 4) to defeat simple 1-to-2-hop heuristic tracking by AML compliance units.
    3. Intermediary Role (intermediary_only=True): A true layering transaction serves as an internal
       conduit, requiring both in-degree >= 1 and out-degree >= 1 (excluding initial wallet sources
       and terminal sink cash-outs).

    Args:
        G: Directed transaction graph (nx.DiGraph) where G.nodes[n]['time'] is timestamp.
        min_path_len: Minimum number of directed edges along the peeling chain (default: 4).
        max_fanout: Maximum out-degree allowed for peeling chain intermediaries (default: 3).
        intermediary_only: If True, requires in_len >= 1 and out_len >= 1 (default: True).
        strict_time: If True, requires time[v] > time[u]. If False, requires time[v] >= time[u].

    Returns:
        Dict[int, bool]: Mapping from node_id to boolean indicator (True = on layering chain).
    """
    labels: Dict[int, bool] = {n: False for n in G.nodes()}

    # Build filtered subgraph with valid temporal ordering and peeling fan-out constraint
    filtered_G = nx.DiGraph()
    filtered_G.add_nodes_from(G.nodes(data=True))

    for u, v in G.edges():
        # Constrain peeling chain transitions by fan-out
        if max_fanout is not None and G.out_degree(u) > max_fanout:
            continue

        t_u = G.nodes[u].get("time", 0)
        t_v = G.nodes[v].get("time", 0)
        if strict_time:
            if t_v > t_u:
                filtered_G.add_edge(u, v)
        else:
            if t_v >= t_u:
                filtered_G.add_edge(u, v)

    if filtered_G.number_of_edges() == 0 and strict_time:
        return find_layering_labels(
            G, min_path_len=min_path_len, max_fanout=max_fanout,
            intermediary_only=intermediary_only, strict_time=False
        )

    try:
        topo_order = list(nx.topological_sort(filtered_G))
        is_dag = True
    except nx.NetworkXUnfeasible:
        is_dag = False

    if is_dag:
        in_len: Dict[int, int] = {n: 0 for n in filtered_G.nodes()}
        out_len: Dict[int, int] = {n: 0 for n in filtered_G.nodes()}

        for n in topo_order:
            for pred in filtered_G.predecessors(n):
                in_len[n] = max(in_len[n], in_len[pred] + 1)

        for n in reversed(topo_order):
            for succ in filtered_G.successors(n):
                out_len[n] = max(out_len[n], out_len[succ] + 1)

        for n in filtered_G.nodes():
            total_len = in_len[n] + out_len[n]
            if total_len >= min_path_len:
                if intermediary_only:
                    if in_len[n] >= 1 and out_len[n] >= 1:
                        labels[n] = True
                else:
                    labels[n] = True
    else:
        # Bounded BFS fallback for non-DAG
        for root in filtered_G.nodes():
            if filtered_G.out_degree(root) == 0:
                continue
            queue = [(root, 0)]
            visited = {root: 0}
            while queue:
                curr, depth = queue.pop(0)
                if depth >= min_path_len:
                    labels[root] = True
                    break
                for nbr in filtered_G.successors(curr):
                    if nbr not in visited or visited[nbr] < depth + 1:
                        visited[nbr] = depth + 1
                        queue.append((nbr, depth + 1))

    return labels


# ==============================================================================
# 3. Smurfing (Structuring) Detection
# ==============================================================================
def find_smurfing_labels(
    G: nx.DiGraph,
    degree_threshold: int = 10,
    time_window: int = 2,
) -> Dict[int, bool]:
    """Identifies nodes exhibiting smurfing / structuring patterns.

    A transaction node is classified as smurfing if its in-degree (fan-in aggregation)
    or out-degree (fan-out distribution) within any `time_window` consecutive time
    steps is >= degree_threshold.

    Args:
        G: Directed transaction graph (nx.DiGraph) where G.nodes[n]['time'] is timestamp.
        degree_threshold: Minimum degree within time window to trigger smurfing flag.
        time_window: Number of consecutive time steps in window.

    Returns:
        Dict[int, bool]: Mapping from node_id to boolean indicator (True = smurfing node).
    """
    labels: Dict[int, bool] = {n: False for n in G.nodes()}

    # Group incoming and outgoing edges by timestamp for each node
    node_times = {n: G.nodes[n].get("time", 0) for n in G.nodes()}

    for n in G.nodes():
        in_deg = G.in_degree(n)
        out_deg = G.out_degree(n)

        # Quick filter: if total degree < threshold, window degree cannot exceed threshold
        if max(in_deg, out_deg) < degree_threshold:
            continue

        # Aggregate degrees by time step
        in_time_counts: Dict[int, int] = defaultdict(int)
        for pred in G.predecessors(n):
            t = node_times.get(pred, node_times[n])
            in_time_counts[t] += 1

        out_time_counts: Dict[int, int] = defaultdict(int)
        for succ in G.successors(n):
            t = node_times.get(n, 0)
            out_time_counts[t] += 1

        # Check sliding windows of size time_window
        all_times = sorted(set(in_time_counts.keys()).union(set(out_time_counts.keys())))
        if not all_times:
            if max(in_deg, out_deg) >= degree_threshold:
                labels[n] = True
            continue

        is_smurf = False
        min_t, max_t = min(all_times), max(all_times)

        for start_t in range(min_t, max_t + 1):
            end_t = start_t + time_window - 1
            in_window = sum(in_time_counts[t] for t in range(start_t, end_t + 1) if t in in_time_counts)
            out_window = sum(out_time_counts[t] for t in range(start_t, end_t + 1) if t in out_time_counts)

            if in_window >= degree_threshold or out_window >= degree_threshold:
                is_smurf = True
                break

        if is_smurf:
            labels[n] = True

    return labels


# ==============================================================================
# 4. Multi-Pattern Label Mining Pipeline
# ==============================================================================
def mine_all_patterns(
    data: Any,
    max_cycle_len: int = 5,
    min_path_len: int = 4,
    max_fanout: int = 3,
    intermediary_only: bool = True,
    degree_threshold: int = 10,
    time_window: int = 2,
    mask_illicit_only: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, torch.Tensor]:
    """Executes the full pattern mining pipeline and returns aligned [N, 3] targets.

    Args:
        data: PyG Data instance or object with x, edge_index, and time.
        max_cycle_len: Maximum cycle length for circular detection.
        min_path_len: Minimum path length for layering detection (peeling chain).
        max_fanout: Maximum out-degree allowed along peeling chain transitions.
        intermediary_only: Require intermediary conduit role for layering.
        degree_threshold: Degree cutoff for smurfing detection.
        time_window: Time window size for smurfing degree aggregation.
        mask_illicit_only: If True, only illicit nodes (y=1) can be labeled positive.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, torch.Tensor]:
            - y_circ: Boolean array [N]
            - y_lay: Boolean array [N]
            - y_smurf: Boolean array [N]
            - pattern_tensor: FloatTensor [N, 3] with columns [circular, layering, smurfing]
    """
    N = data.num_nodes
    print(f"\nBuilding NetworkX DiGraph for {N:,} nodes and {data.num_edges:,} edges...")

    src = data.edge_index[0].cpu().numpy().astype(int)
    dst = data.edge_index[1].cpu().numpy().astype(int)
    node_times = data.time.cpu().numpy().astype(int)

    G = nx.DiGraph()
    G.add_nodes_from(range(N))
    for i in range(N):
        G.nodes[i]["time"] = int(node_times[i])
    for s, d in zip(src, dst):
        G.add_edge(int(s), int(d), time=int(node_times[s]))

    # 1. Circular labels
    print("[1/3] Mining Circular Transfers...")
    circ_dict = find_circular_labels(G, max_cycle_len=max_cycle_len)
    circ_arr = np.array([circ_dict[i] for i in range(N)], dtype=bool)

    # 2. Layering labels
    print("[2/3] Mining Layering (Peeling Chains)...")
    lay_dict = find_layering_labels(
        G, min_path_len=min_path_len, max_fanout=max_fanout,
        intermediary_only=intermediary_only, strict_time=False
    )
    lay_arr = np.array([lay_dict[i] for i in range(N)], dtype=bool)

    # 3. Smurfing labels
    print("[3/3] Mining Smurfing (Structuring)...")
    smurf_dict = find_smurfing_labels(G, degree_threshold=degree_threshold, time_window=time_window)
    smurf_arr = np.array([smurf_dict[i] for i in range(N)], dtype=bool)

    # Intersect with illicit status if requested
    if mask_illicit_only and hasattr(data, "y"):
        is_illicit = (data.y.cpu().numpy() == 1)
        circ_arr = circ_arr & is_illicit
        lay_arr = lay_arr & is_illicit
        smurf_arr = smurf_arr & is_illicit

    # Stack into [N, 3] target tensor: col 0 = circ, col 1 = lay, col 2 = smurf
    multi_label = np.stack([circ_arr, lay_arr, smurf_arr], axis=1).astype(np.float32)
    pattern_tensor = torch.from_numpy(multi_label)

    return circ_arr, lay_arr, smurf_arr, pattern_tensor


# ==============================================================================
# 5. Main Execution
# ==============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("TEMPORAL AML GRAPH TOPOLOGY PATTERN MINING")
    print("=" * 70)

    processed_path = Path("data/processed/elliptic_graph.pt")
    loader = EllipticDatasetLoader()

    if processed_path.exists():
        data = loader.load(processed_path)
    else:
        data = loader.build_graph()
        loader.save()

    y_circ, y_lay, y_smurf, pattern_tensor = mine_all_patterns(
        data,
        max_cycle_len=5,
        min_path_len=4,
        max_fanout=3,
        intermediary_only=True,
        degree_threshold=10,
        time_window=2,
        mask_illicit_only=False,
    )

    print("\n" + "-" * 70)
    print("PATTERN MINING SUMMARY")
    print("-" * 70)
    print(f"Total Graph Nodes : {data.num_nodes:,}")
    print(f"Circular Flagged  : {y_circ.sum():,} ({y_circ.mean() * 100:.2f}%)")
    print(f"Layering Flagged  : {y_lay.sum():,} ({y_lay.mean() * 100:.2f}%)")
    print(f"Smurfing Flagged  : {y_smurf.sum():,} ({y_smurf.mean() * 100:.2f}%)")
    print(f"Stacked Tensor    : {tuple(pattern_tensor.shape)} (dtype: {pattern_tensor.dtype})")

    # Labeled node subset summary
    labeled_mask = (data.y != -1).cpu().numpy()
    illicit_mask = (data.y == 1).cpu().numpy()
    print(f"\nWithin Illicit Nodes ({illicit_mask.sum():,} nodes):")
    print(f"  • Circular : {(y_circ & illicit_mask).sum():,} ({(y_circ & illicit_mask).mean() * 100:.2f}%)")
    print(f"  • Layering : {(y_lay & illicit_mask).sum():,} ({(y_lay & illicit_mask).mean() * 100:.2f}%)")
    print(f"  • Smurfing : {(y_smurf & illicit_mask).sum():,} ({(y_smurf & illicit_mask).mean() * 100:.2f}%)")

    # Save artifact
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)
    save_file = output_dir / "pattern_labels.pt"

    torch.save(
        {
            "patterns": pattern_tensor,
            "y_circ": torch.from_numpy(y_circ),
            "y_lay": torch.from_numpy(y_lay),
            "y_smurf": torch.from_numpy(y_smurf),
        },
        save_file,
    )
    print(f"\n✅ Saved multi-label pattern targets to: {save_file.resolve()}")
    print("=" * 70)
