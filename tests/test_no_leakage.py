"""Unit tests verifying strict temporal causality and absence of lookahead leakage in TGAT."""

import sys
from pathlib import Path
import pytest
import torch

# Ensure workspace root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models import (
    FixedTimeEncoder,
    TemporalNeighborSampler,
    TGATLayer,
    TGATEncoder,
)


@pytest.fixture
def synthetic_graph():
    """Constructs a synthetic temporal transaction graph with known past and future edges.

    Timeline relative to target node 0 at observation time t = 10.0:
    - Node 0: Target at t = 10.0
    - Past edges (t_e < 10.0):
        1 -> 0 at t_e = 2.0
        2 -> 0 at t_e = 5.0
        3 -> 0 at t_e = 9.0
        2 -> 1 at t_e = 1.0 (hop 2 for node 1)
        3 -> 2 at t_e = 4.0 (hop 2 for node 2)
    - Future edges (t_e >= 10.0):
        4 -> 0 at t_e = 15.0 (direct future edge)
        5 -> 0 at t_e = 20.0 (direct future edge)
        4 -> 1 at t_e = 8.0  (future relative to node 1's time 2.0)
    """
    torch.manual_seed(42)
    num_nodes = 6
    feat_dim = 16
    x = torch.randn(num_nodes, feat_dim)

    # Past-only edge set
    past_src = [1, 2, 3, 2, 3]
    past_dst = [0, 0, 0, 1, 2]
    past_times = [2.0, 5.0, 9.0, 1.0, 4.0]

    past_edge_index = torch.tensor([past_src, past_dst], dtype=torch.long)
    past_edge_times = torch.tensor(past_times, dtype=torch.float32)

    # Augmented edge set (Past + Future edges)
    all_src = past_src + [4, 5, 4]
    all_dst = past_dst + [0, 0, 1]
    all_times = past_times + [15.0, 20.0, 8.0]

    all_edge_index = torch.tensor([all_src, all_dst], dtype=torch.long)
    all_edge_times = torch.tensor(all_times, dtype=torch.float32)

    return {
        "x": x,
        "past_edge_index": past_edge_index,
        "past_edge_times": past_edge_times,
        "all_edge_index": all_edge_index,
        "all_edge_times": all_edge_times,
        "target_node": 0,
        "target_time": 10.0,
    }


def test_fixed_time_encoder_properties():
    """Verifies FixedTimeEncoder is non-trainable, uses register_buffer, and matches sinusoidal math."""
    time_dim = 64
    encoder = FixedTimeEncoder(time_dim=time_dim)

    # 1. Assert frequencies are registered as a buffer, NOT nn.Parameter
    param_names = [name for name, _ in encoder.named_parameters()]
    assert len(param_names) == 0, f"FixedTimeEncoder must have 0 trainable parameters, found: {param_names}"

    buffer_names = [name for name, _ in encoder.named_buffers()]
    assert "frequencies" in buffer_names, "frequencies must be registered via register_buffer"

    # 2. Check mathematical values at delta_t = 0
    t0 = torch.tensor([0.0])
    phi_0 = encoder(t0)
    assert phi_0.shape == (1, time_dim)

    # First half are cosines: cos(0) = 1.0
    cos_part = phi_0[0, : time_dim // 2]
    assert torch.allclose(cos_part, torch.ones_like(cos_part))

    # Second half are sines: sin(0) = 0.0
    sin_part = phi_0[0, time_dim // 2 :]
    assert torch.allclose(sin_part, torch.zeros_like(sin_part))


def test_temporal_neighbor_sampler_no_leakage(synthetic_graph):
    """Asserts TemporalNeighborSampler never returns edges where t_e >= t_target, and sorts most-recent-first."""
    all_sampler = TemporalNeighborSampler(
        edge_index=synthetic_graph["all_edge_index"],
        edge_times=synthetic_graph["all_edge_times"],
        num_neighbors=5,
    )

    target_node = synthetic_graph["target_node"]
    target_time = synthetic_graph["target_time"]

    nbr_nodes, nbr_times, mask = all_sampler.sample_one(target_node, target_time)

    # Filter to only valid returned neighbors
    valid_times = nbr_times[mask]
    valid_nodes = nbr_nodes[mask]

    # Strict lookahead assertion: every returned edge must be strictly prior to target_time
    assert len(valid_times) > 0, "Expected valid causal neighbors for node 0"
    assert (valid_times < target_time).all(), (
        f"Temporal leakage detected! Returned edge timestamps: {valid_times.tolist()} for target time {target_time}"
    )

    # Ensure future nodes 4 (t=15) and 5 (t=20) were completely excluded
    assert 4 not in valid_nodes.tolist(), "Future node 4 (t=15) was erroneously sampled!"
    assert 5 not in valid_nodes.tolist(), "Future node 5 (t=20) was erroneously sampled!"

    # Ensure ordering is most-recent-first (descending timestamp order)
    times_list = valid_times.tolist()
    assert times_list == sorted(times_list, reverse=True), (
        f"Neighbors must be sorted most-recent-first: got {times_list}"
    )
    assert times_list == [9.0, 5.0, 2.0], f"Expected times [9.0, 5.0, 2.0], got {times_list}"
    assert valid_nodes.tolist() == [3, 2, 1], f"Expected nodes [3, 2, 1], got {valid_nodes.tolist()}"


def test_temporal_neighbor_sampler_hop2_causality(synthetic_graph):
    """Asserts 2-hop causal consistency: sampling from node 1 at t=2.0 excludes edge (4 -> 1 at t=8.0)."""
    all_sampler = TemporalNeighborSampler(
        edge_index=synthetic_graph["all_edge_index"],
        edge_times=synthetic_graph["all_edge_times"],
        num_neighbors=5,
    )

    # Node 1 at its event time t=2.0
    nbr_nodes, nbr_times, mask = all_sampler.sample_one(node_id=1, time_t=2.0)
    valid_times = nbr_times[mask]
    valid_nodes = nbr_nodes[mask]

    # Edge (4 -> 1 at t=8.0) is in the future relative to t=2.0
    assert (valid_times < 2.0).all()
    assert 4 not in valid_nodes.tolist()
    # Edge (2 -> 1 at t=1.0) is in the past
    assert 2 in valid_nodes.tolist()
    assert 1.0 in valid_times.tolist()


def test_tgat_encoder_causality_invariance(synthetic_graph):
    """Proves zero lookahead: adding future edges (t > t_target) has ZERO effect on node embeddings at t."""
    torch.manual_seed(99)
    x = synthetic_graph["x"]
    target_node = torch.tensor([synthetic_graph["target_node"]])
    target_time = torch.tensor([synthetic_graph["target_time"]])

    # Sampler 1: Graph containing strictly past edges
    sampler_past = TemporalNeighborSampler(
        edge_index=synthetic_graph["past_edge_index"],
        edge_times=synthetic_graph["past_edge_times"],
        num_neighbors=4,
    )

    # Sampler 2: Graph containing past edges PLUS future edges (t=15.0, t=20.0, etc.)
    sampler_with_future = TemporalNeighborSampler(
        edge_index=synthetic_graph["all_edge_index"],
        edge_times=synthetic_graph["all_edge_times"],
        num_neighbors=4,
    )

    # Instantiate 2-layer TGATEncoder
    encoder = TGATEncoder(
        in_dim=16,
        hidden_dim=32,
        time_dim=16,
        num_layers=2,
        num_heads=2,
        max_neighbors=4,
        dropout=0.0,
    )
    encoder.eval()

    with torch.no_grad():
        # Embedding computed using only past graph
        h_past = encoder(x, target_node, target_time, sampler=sampler_past)

        # Embedding computed using graph augmented with future edges
        h_with_future = encoder(x, target_node, target_time, sampler=sampler_with_future)

    # Output dimensions must match
    assert h_past.shape == (1, 32)
    assert h_with_future.shape == (1, 32)

    # Strict mathematical identity: future edges must produce exactly zero difference
    abs_diff = torch.abs(h_past - h_with_future).max().item()
    assert torch.allclose(h_past, h_with_future, atol=1e-6), (
        f"Lookahead leakage detected! Adding future edges changed the embedding by {abs_diff}"
    )


def test_tgat_encoder_batch_consistency(synthetic_graph):
    """Verifies that batch evaluation produces identical embeddings to individual evaluations."""
    torch.manual_seed(99)
    x = synthetic_graph["x"]

    sampler = TemporalNeighborSampler(
        edge_index=synthetic_graph["past_edge_index"],
        edge_times=synthetic_graph["past_edge_times"],
        num_neighbors=4,
    )

    encoder = TGATEncoder(
        in_dim=16,
        hidden_dim=32,
        time_dim=16,
        num_layers=2,
        num_heads=2,
        max_neighbors=4,
        sampler=sampler,
        dropout=0.0,
    )
    encoder.eval()

    nodes = torch.tensor([0, 1])
    times = torch.tensor([10.0, 2.0])

    with torch.no_grad():
        # Batched forward
        h_batch = encoder(x, nodes, times)

        # Individual forwards
        h_node0 = encoder(x, torch.tensor([0]), torch.tensor([10.0]))
        h_node1 = encoder(x, torch.tensor([1]), torch.tensor([2.0]))

    assert torch.allclose(h_batch[0], h_node0[0], atol=1e-6)
    assert torch.allclose(h_batch[1], h_node1[0], atol=1e-6)


def test_sampler_same_timestep_and_descendant_exclusion():
    """Verifies directed causality under same-timestep (<=) sampling.

    Specifically verifies:
    1. Intra-timestep in-edges (t_e == t_target) are properly included.
    2. Topological descendants (nodes connected via out-edges, target -> descendant)
       cannot be sampled into target's receptive field, even when times are equal.
    3. Future in-edges (t_e > t_target) are strictly excluded.
    """
    # Graph structure:
    # 0 -> 1 at t=5.0 (0 is parent of 1)
    # 1 -> 2 at t=5.0 (2 is descendant of 1)
    # 3 -> 1 at t=6.0 (3 is future parent of 1)
    edge_index = torch.tensor([
        [0, 1, 3],  # src
        [1, 2, 1],  # dst
    ], dtype=torch.long)
    edge_times = torch.tensor([5.0, 5.0, 6.0], dtype=torch.float32)

    sampler = TemporalNeighborSampler(
        edge_index=edge_index,
        edge_times=edge_times,
        num_neighbors=5,
    )

    # 1. Query Target Node 1 at t = 5.0
    nbrs_1, times_1, mask_1 = sampler.sample_one(node_id=1, time_t=5.0)
    valid_nbrs_1 = nbrs_1[mask_1].tolist()
    valid_times_1 = times_1[mask_1].tolist()

    # Parent 0 must be sampled (in-edge at t=5.0 <= 5.0)
    assert 0 in valid_nbrs_1, "Parent node 0 (t=5.0) must be included under causal <= sampling"
    assert 5.0 in valid_times_1

    # Descendant 2 (out-edge 1 -> 2) must NEVER be sampled into node 1's receptive field
    assert 2 not in valid_nbrs_1, (
        "Topological descendant node 2 was sampled into target node 1! "
        "Sampler must only follow directed in-edges."
    )

    # Future parent 3 (t=6.0 > 5.0) must NEVER be sampled
    assert 3 not in valid_nbrs_1, "Future parent node 3 (t=6.0 > 5.0) was erroneously sampled!"

    assert valid_nbrs_1 == [0], f"Expected valid neighbors for node 1 to be [0], got {valid_nbrs_1}"

    # 2. Query Ancestor Node 0 at t = 5.0 (has 0 in-edges)
    nbrs_0, times_0, mask_0 = sampler.sample_one(node_id=0, time_t=5.0)
    assert mask_0.sum().item() == 0, "Node 0 has no incoming edges and must return empty neighborhood"

    # 3. Query Descendant Node 2 at t = 5.0 (has in-edge 1 -> 2 at t=5.0)
    nbrs_2, times_2, mask_2 = sampler.sample_one(node_id=2, time_t=5.0)
    assert nbrs_2[mask_2].tolist() == [1], "Descendant node 2 should sample its parent node 1"


def test_two_hop_causal_anchoring():
    """Verifies that 2-hop neighbor expansion strictly anchors its causal cutoff to the

    intermediate hop-1 node's event timestamp, NEVER borrowing the later target node's timestamp.

    Topology:
        Node 0 (Target i): evaluated at t_i = 8.0
        Node 1 (Hop-1 j):  edge 1 -> 0 at t_j = 5.0 (<= 8.0, valid hop-1)
        Node 2 (Hop-2 k):  edge 2 -> 1 at t_k = 5.0 (<= 5.0, valid hop-2)
        Node 3 (Hop-2 m):  edge 3 -> 1 at t_m = 7.0
                           Notice: t_m (7.0) <= t_i (8.0), BUT t_m (7.0) > t_j (5.0).
                           If Hop-2 borrows target 0's cutoff, node 3 would leak into the receptive field.
                           Because Hop-2 is causally anchored to t_j = 5.0, node 3 must be strictly excluded.
    """
    torch.manual_seed(42)
    feat_dim = 16
    x = torch.randn(4, feat_dim)

    edge_index = torch.tensor([
        [1, 2, 3],  # src: j=1 -> i=0; k=2 -> j=1; m=3 -> j=1
        [0, 1, 1],  # dst
    ], dtype=torch.long)
    edge_times = torch.tensor([5.0, 5.0, 7.0], dtype=torch.float32)

    sampler = TemporalNeighborSampler(
        edge_index=edge_index,
        edge_times=edge_times,
        num_neighbors=5,
    )

    # 1. Direct Sampler Verification on Hop-2 reference timestamp
    # Hop 1 of target 0 at t=8.0 returns node 1 at time 5.0
    nbrs_1, times_1, mask_1 = sampler.sample_one(node_id=0, time_t=8.0)
    valid_h1_nodes = nbrs_1[mask_1].tolist()
    valid_h1_times = times_1[mask_1].tolist()
    assert valid_h1_nodes == [1], f"Hop 1 must sample node 1, got {valid_h1_nodes}"
    assert valid_h1_times == [5.0]

    # Hop 2 MUST expand node 1 at its own event time t_1 = 5.0, NOT target's time t_0 = 8.0
    h1_node = valid_h1_nodes[0]
    h1_event_time = valid_h1_times[0]
    nbrs_2, times_2, mask_2 = sampler.sample_one(node_id=h1_node, time_t=h1_event_time)
    valid_h2_nodes = nbrs_2[mask_2].tolist()

    # Node 2 (t=5.0 <= 5.0) must be present
    assert 2 in valid_h2_nodes, "Hop-2 node 2 (t=5.0 <= 5.0) must be included"

    # Node 3 (t=7.0 > 5.0) must NEVER be present, despite 7.0 <= 8.0
    assert 3 not in valid_h2_nodes, (
        "Temporal lookahead leakage in hop 2! Node 3 (t=7.0) was sampled for node 1 (t=5.0) "
        "because it erroneously borrowed target node 0's later timestamp (t=8.0)."
    )
    assert valid_h2_nodes == [2], f"Expected hop-2 to contain only [2], got {valid_h2_nodes}"

    # 2. End-to-End TGATEncoder Embedding Invariance Verification
    encoder = TGATEncoder(
        in_dim=feat_dim,
        hidden_dim=32,
        time_dim=16,
        num_layers=2,
        num_heads=2,
        max_neighbors=5,
        sampler=sampler,
        dropout=0.0,
    )
    encoder.eval()

    target_node = torch.tensor([0])
    target_time = torch.tensor([8.0])

    with torch.no_grad():
        h_baseline = encoder(x, target_node, target_time)

        # Perturb Node 3 (the future hop-2 node relative to node 1)
        x_perturbed_m = x.clone()
        x_perturbed_m[3] += 100.0  # Massive perturbation
        h_perturbed_m = encoder(x_perturbed_m, target_node, target_time)

        # Strict mathematical identity: Node 3 has ZERO effect on target 0's embedding
        diff_m = torch.abs(h_baseline - h_perturbed_m).max().item()
        assert diff_m == 0.0, (
            f"Node 3 leaked into target 0's embedding! Embedding changed by {diff_m} "
            "when modifying future hop-2 node 3."
        )

        # Perturb Node 2 (the valid causal hop-2 node)
        x_perturbed_k = x.clone()
        x_perturbed_k[2] += 10.0
        h_perturbed_k = encoder(x_perturbed_k, target_node, target_time)

        # Node 2 DOES reach target 0's receptive field
        diff_k = torch.abs(h_baseline - h_perturbed_k).max().item()
        assert diff_k > 1e-5, "Valid causal hop-2 node 2 had no effect on target embedding!"


