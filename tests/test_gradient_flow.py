"""Unit tests verifying gradient flow through LearnableFourierTimeEncoder and parameter updates in TGAT."""

import sys
from pathlib import Path
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure workspace root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models import (
    FixedTimeEncoder,
    LearnableFourierTimeEncoder,
    TemporalNeighborSampler,
    TGATEncoder,
)


def test_learnable_fourier_time_encoder_shapes():
    """Verifies that LearnableFourierTimeEncoder supports arbitrary batch shapes and registers omega as parameter."""
    time_dim = 16
    encoder = LearnableFourierTimeEncoder(time_dim=time_dim, initial_scale=0.1)

    # 1. Parameter verification
    params = list(encoder.named_parameters())
    param_dict = dict(params)
    assert "omega" in param_dict, "omega must be an nn.Parameter"
    assert param_dict["omega"].shape == (time_dim // 2,)
    assert param_dict["omega"].requires_grad is True

    # 2. 1D input [B]
    t_1d = torch.tensor([1.0, 3.5, 7.0])
    out_1d = encoder(t_1d)
    assert out_1d.shape == (3, time_dim)

    # 3. 2D input [B, M]
    t_2d = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    out_2d = encoder(t_2d)
    assert out_2d.shape == (2, 3, time_dim)

    # 4. 3D input [B, N, M]
    t_3d = torch.randn(2, 3, 4).abs()
    out_3d = encoder(t_3d)
    assert out_3d.shape == (2, 3, 4, time_dim)


def test_tgat_encoder_time_encoder_type_instantiation():
    """Verifies that TGATEncoder correctly instantiates fixed vs learnable time encoders based on flag."""
    # Fixed encoder
    fixed_encoder = TGATEncoder(
        in_dim=8,
        hidden_dim=16,
        time_dim=8,
        time_encoder_type="fixed",
    )
    assert isinstance(fixed_encoder.time_encoder, FixedTimeEncoder)
    assert "omega" not in dict(fixed_encoder.time_encoder.named_parameters())

    # Learnable encoder
    learnable_encoder = TGATEncoder(
        in_dim=8,
        hidden_dim=16,
        time_dim=8,
        time_encoder_type="learnable",
        initial_scale=0.05,
    )
    assert isinstance(learnable_encoder.time_encoder, LearnableFourierTimeEncoder)
    assert "omega" in dict(learnable_encoder.time_encoder.named_parameters())


def test_tgat_gradient_flow_and_omega_update():
    """Builds a tiny TGATEncoder with LearnableFourierTimeEncoder, runs forward+backward on binary classification loss,

    and asserts omega.grad is not None, not all zero, finite, and omega changes after an optimizer step.
    """
    torch.manual_seed(42)

    # Synthetic graph with 4 nodes
    # Node 0 at time t=10.0 with incoming causal edges from 1, 2, 3
    src = [1, 2, 3]
    dst = [0, 0, 0]
    edge_times = [2.0, 5.0, 9.0]
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_times_tensor = torch.tensor(edge_times, dtype=torch.float32)

    sampler = TemporalNeighborSampler(
        edge_index=edge_index,
        edge_times=edge_times_tensor,
        num_neighbors=3,
    )

    in_dim = 8
    hidden_dim = 16
    time_dim = 8

    # Instantiate TGATEncoder with learnable time encoder
    encoder = TGATEncoder(
        in_dim=in_dim,
        hidden_dim=hidden_dim,
        time_dim=time_dim,
        num_layers=2,
        num_heads=2,
        max_neighbors=3,
        time_encoder_type="learnable",
        initial_scale=0.1,
        sampler=sampler,
        dropout=0.0,
    )

    # Binary classification head
    classifier_head = nn.Linear(hidden_dim, 1)

    # Optimizer covering encoder (including learnable omega) and classification head
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(classifier_head.parameters()),
        lr=0.05,
    )

    x = torch.randn(4, in_dim)
    target_nodes = torch.tensor([0])
    target_times = torch.tensor([10.0])
    target_labels = torch.tensor([1.0])  # Binary ground truth

    # Forward pass
    optimizer.zero_grad()
    h = encoder(x, target_nodes, target_times)
    logits = classifier_head(h).squeeze(-1)
    loss = F.binary_cross_entropy_with_logits(logits, target_labels)

    # Backward pass
    loss.backward()

    # 1. Assert omega.grad is not None
    omega = encoder.time_encoder.omega
    assert omega.grad is not None, "time_encoder.omega.grad must NOT be None after backward pass!"

    grad = omega.grad

    # 2. Assert omega.grad is not all zero
    assert not torch.all(grad == 0.0), "time_encoder.omega.grad must NOT be all zeros (gradients must flow to omega)!"

    # 3. Assert no NaN or Inf values
    assert not torch.isnan(grad).any(), "time_encoder.omega.grad contains NaN values!"
    assert not torch.isinf(grad).any(), "time_encoder.omega.grad contains Inf values!"

    # 4. Record omega before step and assert it actually updates after optimizer.step()
    omega_before = omega.clone().detach()
    optimizer.step()
    omega_after = omega.detach()

    # Verify that omega parameter has actually been modified
    assert not torch.allclose(omega_before, omega_after), (
        f"omega parameters failed to update after optimizer.step(): before={omega_before}, after={omega_after}"
    )

    # Verify that each frequency coordinate in omega moved
    diff = torch.abs(omega_before - omega_after)
    assert (diff > 0).all(), f"Every coordinate in omega should receive an update, got diff: {diff}"
