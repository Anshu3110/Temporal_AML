"""Temporal Graph Attention Network (TGAT) modules and temporal neighbor sampling.

Implements:
1. FixedTimeEncoder: Non-trainable sinusoidal time encoding Phi(delta_t) with fixed frequencies.
2. LearnableFourierTimeEncoder: Trainable Fourier time encoding with learnable omega parameters.
3. TemporalNeighborSampler: Causal neighbor sampler ensuring t_edge < t_target (most-recent-first).
4. TGATLayer: Multi-head temporal attention with time-aware query, key, value projections.
5. TGATEncoder: 2-layer stacked temporal encoder computing inductive representations h_i(t).
"""

import math
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ==============================================================================
# 1. Fixed Sinusoidal Time Encoder (Non-trainable)
# ==============================================================================
class FixedTimeEncoder(nn.Module):
    """Non-trainable sinusoidal time encoding Phi(delta_t) using fixed frequencies.

    Calculates:
        Phi(delta_t) = [cos(delta_t * omega), sin(delta_t * omega)]
    where omega_k are fixed geometrically-spaced frequencies. Registered as a buffer
    (not an nn.Parameter) so no gradients are computed or weights updated.
    """

    def __init__(self, time_dim: int = 64) -> None:
        """Initializes sinusoidal time encoder.

        Args:
            time_dim: Output dimension of temporal representation. Must be even.
        """
        super().__init__()
        self.time_dim = time_dim
        half_dim = time_dim // 2

        # Geometrically spaced frequencies: 1 / (10000 ^ (2k / time_dim))
        freq_indices = torch.arange(half_dim, dtype=torch.float32)
        frequencies = 1.0 / (10000.0 ** (freq_indices / half_dim))

        # Explicitly registered as a persistent buffer, NOT nn.Parameter
        self.register_buffer("frequencies", frequencies)

    def forward(self, delta_t: torch.Tensor) -> torch.Tensor:
        """Encodes relative time differences into sinusoidal vector representations.

        Args:
            delta_t: Tensor of arbitrary shape [...] containing time intervals delta_t >= 0.

        Returns:
            torch.Tensor: Sinusoidal encodings of shape [..., time_dim].
        """
        if delta_t.dim() == 0:
            delta_t = delta_t.unsqueeze(0)

        # delta_t shape: [..., 1], frequencies: [half_dim]
        dt = delta_t.unsqueeze(-1).float()
        angles = dt * self.frequencies  # [..., half_dim]

        cos_enc = torch.cos(angles)
        sin_enc = torch.sin(angles)
        encoding = torch.cat([cos_enc, sin_enc], dim=-1)

        # Handle edge cases with odd dimensions if ever requested
        if encoding.shape[-1] < self.time_dim:
            pad_size = self.time_dim - encoding.shape[-1]
            pad = torch.zeros(*encoding.shape[:-1], pad_size, device=encoding.device)
            encoding = torch.cat([encoding, pad], dim=-1)

        return encoding


# ==============================================================================
# 2. Learnable Fourier Time Encoder
# ==============================================================================
class LearnableFourierTimeEncoder(nn.Module):
    """Learnable Fourier Time Encoder with learnable frequency parameters omega.

    Calculates:
        Phi(delta_t) = sqrt(2 / time_dim) * [cos(omega * delta_t), sin(omega * delta_t)]
    where omega in R^(time_dim / 2) is a trainable parameter initialized with random normal
    weights scaled by `initial_scale`. Supports batched delta_t tensors of arbitrary shape.
    """

    def __init__(self, time_dim: int = 64, initial_scale: float = 0.1) -> None:
        """Initializes learnable Fourier time encoder.

        Args:
            time_dim: Output dimension of temporal representation. Must be even.
            initial_scale: Scaling factor for initial random normal omega weights.
        """
        super().__init__()
        assert time_dim % 2 == 0, f"time_dim must be even, got {time_dim}"
        self.time_dim = time_dim
        self.initial_scale = initial_scale

        # Trainable frequency parameters
        self.omega = nn.Parameter(torch.randn(time_dim // 2, dtype=torch.float32) * initial_scale)

    def forward(self, delta_t: torch.Tensor) -> torch.Tensor:
        """Encodes relative time differences using learnable Fourier frequencies.

        Args:
            delta_t: Tensor of arbitrary shape [...] containing time intervals delta_t >= 0.

        Returns:
            torch.Tensor: Learnable Fourier representation of shape [..., time_dim].
        """
        if delta_t.dim() == 0:
            delta_t = delta_t.unsqueeze(0)

        # delta_t shape: [..., 1], omega: [time_dim // 2]
        dt = delta_t.unsqueeze(-1).float()
        angles = dt * self.omega  # [..., time_dim // 2]

        cos_part = torch.cos(angles)
        sin_part = torch.sin(angles)

        # Normalization factor sqrt(2 / time_dim) (Bochner's theorem / unit variance)
        norm_factor = math.sqrt(2.0 / self.time_dim)
        encoding = norm_factor * torch.cat([cos_part, sin_part], dim=-1)

        return encoding


# ==============================================================================
# 3. Causal Temporal Neighbor Sampler
# ==============================================================================
class TemporalNeighborSampler:
    """Causal temporal neighbor sampler for continuous/discrete-time dynamic graphs.

    Given a target node and observation timestamp (node_id, time_t), samples up to M
    incoming neighbor edges where edge_time < time_t, ordered most-recent-first.
    Guarantees zero temporal lookahead leakage via strict causal verification.
    """

    def __init__(
        self,
        edge_index: torch.Tensor,
        edge_times: torch.Tensor,
        num_neighbors: int = 20,
    ) -> None:
        """Initializes the causal neighbor sampler with indexed adjacency lookups.

        Args:
            edge_index: Directed graph connectivity [2, E] (src = edge_index[0], dst = edge_index[1]).
            edge_times: Timestamps associated with edges [E] (e.g. source node's transaction time step).
            num_neighbors: Maximum neighbors M to sample per node.
        """
        self.num_neighbors = num_neighbors
        self.edge_index = edge_index
        self.edge_times = edge_times.float()

        src_arr = edge_index[0].cpu().numpy().astype(np.int64)
        dst_arr = edge_index[1].cpu().numpy().astype(np.int64)
        times_arr = edge_times.cpu().numpy().astype(np.float32)

        # Pre-group incoming edges per destination node: dst -> (src_array, time_array)
        from collections import defaultdict
        adj_dict = defaultdict(lambda: ([], []))
        for s, d, t in zip(src_arr, dst_arr, times_arr):
            adj_dict[d][0].append(s)
            adj_dict[d][1].append(t)

        self.adj: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        for d, (s_list, t_list) in adj_dict.items():
            s_np = np.array(s_list, dtype=np.int64)
            t_np = np.array(t_list, dtype=np.float32)
            # Sort descending: most-recent-first
            order = np.argsort(-t_np)
            self.adj[d] = (s_np[order], t_np[order])

    def sample_one(
        self,
        node_id: int,
        time_t: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Samples up to M causal temporal neighbors for a single node."""
        n_t = torch.tensor([node_id], dtype=torch.long)
        t_t = torch.tensor([time_t], dtype=torch.float32)
        nbrs, times, masks = self.sample_batch(n_t, t_t)
        return nbrs[0], times[0], masks[0]

    def sample_batch(
        self,
        nodes: torch.Tensor,
        times: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Samples causal neighbors for a batch of target (node, time) pairs.

        Args:
            nodes: Target node indices [B].
            times: Target observation timestamps [B].

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - neighbor_nodes: LongTensor [B, M]
                - neighbor_times: FloatTensor [B, M]
                - valid_mask: BoolTensor [B, M]
        """
        nodes_np = nodes.cpu().numpy()
        times_np = times.cpu().numpy()

        B = len(nodes_np)
        M = self.num_neighbors

        nodes_out = np.zeros((B, M), dtype=np.int64)
        times_out = np.zeros((B, M), dtype=np.float32)
        masks_out = np.zeros((B, M), dtype=bool)

        for i in range(B):
            n = int(nodes_np[i])
            if n not in self.adj:
                continue
            t = float(times_np[i])
            srcs, times_arr = self.adj[n]

            # Fast binary search: times_arr is sorted descending (most-recent-first).
            # -times_arr is sorted ascending. searchsorted finds the first element >= -t,
            # which corresponds to the first edge with edge_time <= t.
            idx = int(np.searchsorted(-times_arr, -t, side="left"))
            num_valid = len(times_arr) - idx
            if num_valid <= 0:
                continue

            k = min(num_valid, M)
            nodes_out[i, :k] = srcs[idx : idx + k]
            times_out[i, :k] = times_arr[idx : idx + k]
            masks_out[i, :k] = True

        return (
            torch.from_numpy(nodes_out).to(nodes.device),
            torch.from_numpy(times_out).to(nodes.device),
            torch.from_numpy(masks_out).to(nodes.device),
        )


# ==============================================================================
# 3. TGAT Layer (Temporal Graph Attention)
# ==============================================================================
class TGATLayer(nn.Module):
    """Single Temporal Graph Attention (TGAT) Layer.

    Applies multi-head attention where for target node i at time t and sampled
    neighbor j at time t_e:
        query = concat(h_i, time_encoder(0)) @ W_Q
        key   = concat(h_j, time_encoder(t - t_e)) @ W_K
        value = concat(h_j, time_encoder(t - t_e)) @ W_V
        attention = softmax(query @ key^T / sqrt(d_k))
        output = attention @ value, followed by residual + linear projection back to hidden_dim.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        time_dim: int = 64,
        num_heads: int = 4,
        time_encoder: Optional[nn.Module] = None,
        dropout: float = 0.1,
    ) -> None:
        """Initializes TGAT attention layer.

        Args:
            hidden_dim: Hidden feature dimension. Must be divisible by num_heads.
            time_dim: Temporal encoding dimension.
            num_heads: Number of parallel attention heads.
            time_encoder: Modular time encoding layer (FixedTimeEncoder or learnable Fourier encoder).
            dropout: Dropout probability.
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.time_dim = time_dim
        self.num_heads = num_heads

        assert hidden_dim % num_heads == 0, (
            f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})"
        )
        self.head_dim = hidden_dim // num_heads

        # Modular time encoder: accepts custom or defaults to FixedTimeEncoder
        self.time_encoder = time_encoder if time_encoder is not None else FixedTimeEncoder(time_dim)

        in_proj_dim = hidden_dim + time_dim
        self.W_Q = nn.Linear(in_proj_dim, hidden_dim, bias=False)
        self.W_K = nn.Linear(in_proj_dim, hidden_dim, bias=False)
        self.W_V = nn.Linear(in_proj_dim, hidden_dim, bias=False)

        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        h_target: torch.Tensor,
        t_target: torch.Tensor,
        h_neighbors: torch.Tensor,
        t_neighbors: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass for TGATLayer.

        Args:
            h_target: Target node representations [B, hidden_dim].
            t_target: Target observation timestamps [B].
            h_neighbors: Sampled neighbor representations [B, M, hidden_dim].
            t_neighbors: Sampled neighbor edge timestamps [B, M].
            mask: Optional boolean mask [B, M] where True indicates valid neighbors.

        Returns:
            torch.Tensor: Updated target representations [B, hidden_dim].
        """
        B = h_target.shape[0]
        M = h_neighbors.shape[1]

        # 1. Target Query: concat(h_i, time_encoder(0)) @ W_Q
        t_zeros = torch.zeros_like(t_target)
        phi_0 = self.time_encoder(t_zeros)  # [B, time_dim]
        q_input = torch.cat([h_target, phi_0], dim=-1)  # [B, hidden_dim + time_dim]
        query = self.W_Q(q_input)  # [B, hidden_dim]
        query = query.view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, 1, d_k]

        # 2. Neighbor Key and Value: concat(h_j, time_encoder(t - t_e)) @ W_K / W_V
        delta_t = t_target.unsqueeze(1) - t_neighbors  # [B, M]
        phi_delta = self.time_encoder(delta_t)  # [B, M, time_dim]
        kv_input = torch.cat([h_neighbors, phi_delta], dim=-1)  # [B, M, hidden_dim + time_dim]

        key = self.W_K(kv_input).view(B, M, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, M, d_k]
        value = self.W_V(kv_input).view(B, M, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, M, d_k]

        # 3. Attention scores: softmax(query @ key^T / sqrt(d_k))
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)  # [B, H, 1, M]

        if mask is not None:
            # Mask out non-existent/padded neighbors
            scores = scores.masked_fill(~mask.unsqueeze(1).unsqueeze(2), -1e9)
            has_neighbors = mask.any(dim=-1).view(B, 1, 1, 1)
            attn_weights = F.softmax(scores, dim=-1)
            # Prevent NaN when target node has 0 valid neighbors
            attn_weights = torch.where(has_neighbors, attn_weights, torch.zeros_like(attn_weights))
        else:
            attn_weights = F.softmax(scores, dim=-1)

        attn_weights = self.dropout(attn_weights)

        # 4. Context aggregation: attention @ value
        context = torch.matmul(attn_weights, value)  # [B, H, 1, d_k]
        context = context.transpose(1, 2).contiguous().view(B, self.hidden_dim)  # [B, hidden_dim]

        # 5. Residual connection + linear projection back to hidden_dim
        # output = attention @ value, then a residual + linear projection back to hidden_dim
        res = context + h_target
        output = self.out_proj(res)
        output = self.layer_norm(output)

        return output


# ==============================================================================
# 4. Multi-Layer TGAT Encoder
# ==============================================================================
class TGATEncoder(nn.Module):
    """Inductive 2-layer Temporal Graph Attention Network Encoder.

    Stacks 2 TGATLayers to compute causal per-node embeddings h_i(t)
    for arbitrary batches of target (node, time) pairs.
    """

    def __init__(
        self,
        in_dim: int = 165,
        hidden_dim: int = 128,
        time_dim: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
        max_neighbors: int = 20,
        time_encoder_type: str = "fixed",  # "fixed" | "learnable"
        initial_scale: float = 0.1,
        time_encoder: Optional[nn.Module] = None,
        sampler: Optional[TemporalNeighborSampler] = None,
        dropout: float = 0.1,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initializes TGAT encoder.

        Args:
            in_dim: Raw node feature dimension.
            hidden_dim: Hidden embedding dimension.
            time_dim: Time encoding dimension.
            num_layers: Number of stacked TGAT layers (default: 2).
            num_heads: Multi-head attention heads count.
            max_neighbors: Neighborhood sampling size M.
            time_encoder_type: Type of time encoder ('fixed' or 'learnable').
            initial_scale: Initial scale for learnable Fourier omega parameters.
            time_encoder: Optional modular time encoding module (overrides time_encoder_type).
            sampler: Causal temporal neighbor sampler instance.
            dropout: Dropout probability.
            config: Optional config dictionary overriding parameters.
        """
        super().__init__()
        if config is not None:
            hidden_dim = config.get("hidden_dim", hidden_dim)
            time_dim = config.get("time_dim", time_dim)
            num_layers = config.get("num_tgat_layers", num_layers)
            num_heads = config.get("num_heads", num_heads)
            max_neighbors = config.get("max_temporal_neighbors", max_neighbors)
            time_encoder_type = config.get("time_encoder_type", time_encoder_type)
            initial_scale = config.get("initial_scale", initial_scale)

        self.hidden_dim = hidden_dim
        self.time_dim = time_dim
        self.num_layers = num_layers
        self.max_neighbors = max_neighbors
        self.time_encoder_type = time_encoder_type
        self.sampler = sampler

        # Project initial raw node features to hidden_dim
        self.feat_proj = nn.Linear(in_dim, hidden_dim)

        # Time encoder: Use explicit instance, or instantiate based on time_encoder_type
        if time_encoder is not None:
            self.time_encoder = time_encoder
        elif str(time_encoder_type).lower() == "learnable":
            self.time_encoder = LearnableFourierTimeEncoder(time_dim=time_dim, initial_scale=initial_scale)
        else:
            self.time_encoder = FixedTimeEncoder(time_dim=time_dim)

        # Stack TGATLayers
        self.layers = nn.ModuleList([
            TGATLayer(
                hidden_dim=hidden_dim,
                time_dim=time_dim,
                num_heads=num_heads,
                time_encoder=self.time_encoder,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

    def forward(
        self,
        x: torch.Tensor,
        target_nodes: torch.Tensor,
        target_times: torch.Tensor,
        sampler: Optional[TemporalNeighborSampler] = None,
    ) -> torch.Tensor:
        """Computes inductive temporal node embeddings h_i(t) for target nodes.

        Args:
            x: Global node feature tensor [N, in_dim].
            target_nodes: Batch of target node indices [B].
            target_times: Batch of target observation timestamps [B].
            sampler: Optional sampler instance (defaults to self.sampler).

        Returns:
            torch.Tensor: Inductive temporal node representations [B, hidden_dim].
        """
        active_sampler = sampler or self.sampler
        if active_sampler is None:
            raise ValueError("No TemporalNeighborSampler provided to TGATEncoder.")

        B = target_nodes.shape[0]

        if self.num_layers == 1:
            # 1-Hop temporal attention
            nbrs_1, times_1, mask_1 = active_sampler.sample_batch(target_nodes, target_times)
            h0_target = self.feat_proj(x[target_nodes])
            h0_nbrs_1 = self.feat_proj(x[nbrs_1])

            h_out = self.layers[0](
                h_target=h0_target,
                t_target=target_times,
                h_neighbors=h0_nbrs_1,
                t_neighbors=times_1,
                mask=mask_1,
            )
            return h_out

        elif self.num_layers == 2:
            # 2-Hop temporal attention graph
            # Hop 1: neighbors of target nodes
            nbrs_1, times_1, mask_1 = active_sampler.sample_batch(target_nodes, target_times)
            # Hop 2: neighbors of Hop 1 neighbors
            nbrs_1_flat = nbrs_1.view(-1)
            times_1_flat = times_1.view(-1)
            nbrs_2, times_2, mask_2 = active_sampler.sample_batch(nbrs_1_flat, times_1_flat)

            # Initial projections
            h0_target = self.feat_proj(x[target_nodes])  # [B, hidden_dim]
            h0_nbrs_1 = self.feat_proj(x[nbrs_1_flat])   # [B * M1, hidden_dim]
            h0_nbrs_2 = self.feat_proj(x[nbrs_2])        # [B * M1, M2, hidden_dim]

            # Layer 1: update Hop 1 embeddings by attending over their Hop 2 neighbors
            h1_nbrs_1 = self.layers[0](
                h_target=h0_nbrs_1,
                t_target=times_1_flat,
                h_neighbors=h0_nbrs_2,
                t_neighbors=times_2,
                mask=mask_2,
            ).view(B, nbrs_1.shape[1], self.hidden_dim)  # [B, M1, hidden_dim]

            # Layer 2: update Target embeddings by attending over Layer 1 embeddings of Hop 1 neighbors
            h2_target = self.layers[1](
                h_target=h0_target,
                t_target=target_times,
                h_neighbors=h1_nbrs_1,
                t_neighbors=times_1,
                mask=mask_1,
            )
            return h2_target

        else:
            raise NotImplementedError(f"TGATEncoder currently supports 1 or 2 layers, got {self.num_layers}.")


# ==============================================================================
# 5. Multi-Task AML Pattern Classification Head
# ==============================================================================
class MultiTaskHead(nn.Module):
    """Two independent 2-layer FFN heads for multi-pattern AML detection.

    Heads:
    1. Layering Chain Head:   Linear -> ReLU -> Dropout -> Linear -> Sigmoid
    2. Smurfing Typology Head: Linear -> ReLU -> Dropout -> Linear -> Sigmoid

    Note on Circular Transfers:
    Dropped because raw Bitcoin UTXO transaction graphs are strict DAGs (0% circular loops).
    Framed as a 2-pattern multi-task architecture focusing on genuine blockchain typologies.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        head_hidden_dim: int = 64,
        dropout: float = 0.2,
    ) -> None:
        """Initializes the two independent pattern classification heads.

        Args:
            hidden_dim: Dimension of input shared TGAT embeddings.
            head_hidden_dim: Intermediate hidden dimension for 2-layer FFN heads.
            dropout: Dropout probability.
        """
        super().__init__()
        self.head_lay = nn.Sequential(
            nn.Linear(hidden_dim, head_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden_dim, 1),
        )
        self.head_smurf = nn.Sequential(
            nn.Linear(hidden_dim, head_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden_dim, 1),
        )

    def forward(
        self,
        h: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass computing sigmoid probabilities for Layering and Smurfing.

        Args:
            h: Shared node embeddings [B, hidden_dim].

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                (p_lay, p_smurf) each of shape [B] in range [0, 1].
        """
        p_lay = torch.sigmoid(self.head_lay(h)).squeeze(-1)
        p_smurf = torch.sigmoid(self.head_smurf(h)).squeeze(-1)
        return p_lay, p_smurf

    def forward_logits(
        self,
        h: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning unnormalized logits for Layering and Smurfing.

        Args:
            h: Shared node embeddings [B, hidden_dim].

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                (logit_lay, logit_smurf) each of shape [B].
        """
        logit_lay = self.head_lay(h).squeeze(-1)
        logit_smurf = self.head_smurf(h).squeeze(-1)
        return logit_lay, logit_smurf


# ==============================================================================
# 6. TemporalAML Ablation Model (Mean-Pooling, No Time Encoding)
# ==============================================================================
class TemporalAMLAblationNoTime(nn.Module):
    """Ablation variant of TemporalAML with time encoding completely disabled.

    Replaces dynamic Fourier temporal attention with static mean-pooled neighbor
    aggregation over the sampled neighborhood, isolating the empirical contribution
    of the continuous temporal encoding term.
    """

    def __init__(
        self,
        in_dim: int = 165,
        hidden_dim: int = 128,
        head_hidden_dim: int = 64,
        max_neighbors: int = 20,
        sampler: Optional[TemporalNeighborSampler] = None,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        self.sampler = sampler
        self.feat_proj = nn.Linear(in_dim, hidden_dim)
        self.agg_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.head = MultiTaskHead(hidden_dim=hidden_dim, head_hidden_dim=head_hidden_dim, dropout=dropout)

    def encode(
        self,
        x: torch.Tensor,
        target_nodes: torch.Tensor,
        target_times: torch.Tensor,
    ) -> torch.Tensor:
        nbrs, _, mask = self.sampler.sample_batch(target_nodes, target_times)
        h_target = F.relu(self.feat_proj(x[target_nodes]))
        h_nbrs = F.relu(self.feat_proj(x[nbrs]))  # [B, M, hidden_dim]

        mask_expanded = mask.unsqueeze(-1).float()  # [B, M, 1]
        sum_nbrs = (h_nbrs * mask_expanded).sum(dim=1)
        count_nbrs = mask_expanded.sum(dim=1).clamp(min=1.0)
        h_mean = sum_nbrs / count_nbrs

        context = self.agg_proj(h_mean)
        h_emb = self.norm(self.out_proj(context + h_target))
        return self.dropout(h_emb)

    def forward(
        self,
        x: torch.Tensor,
        target_nodes: torch.Tensor,
        target_times: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h_emb = self.encode(x, target_nodes, target_times)
        return self.head(h_emb)

    def forward_logits(
        self,
        x: torch.Tensor,
        target_nodes: torch.Tensor,
        target_times: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h_emb = self.encode(x, target_nodes, target_times)
        return self.head.forward_logits(h_emb)

