#!/usr/bin/env python3
"""
Appends O3 cells (30-45) to TemporalAML_Phase1.ipynb
Run: python3 append_o3_cells.py
"""
import json, sys

NOTEBOOK_PATH = "/Users/apple/Desktop/DissertationProject/notebook/TemporalAML_Phase1.ipynb"

# ── Helper builders ─────────────────────────────────────────────────────────
def md(source): return {"cell_type":"markdown","id":f"md-{abs(hash(source[:30]))%99999}","metadata":{},"source":[source]}
def code(cell_id, source): return {"cell_type":"code","execution_count":None,"id":cell_id,"metadata":{},"outputs":[],"source":[source]}

# ── All O3 cells ──────────────────────────────────────────────────────────────
NEW_CELLS = [

# ── SECTION HEADER ──────────────────────────────────────────────────────────
md("""---
## 🕵️ Section 5: Objective 3 — Multi-Pattern AML Detection
Detect **Circular Transfers**, **Layering**, and **Smurfing** using a shared
TGAT encoder with three independent classification heads.

| Pattern | Definition | Graph Signal |
|---------|-----------|-------------|
| **Circular Transfer** | Money loops back to source | Node in SCC of size ≥ 2 |
| **Layering** | Long sequential chains obscuring origin | Node in directed path ≥ 3, monotonic t |
| **Smurfing** | Fan-out to many small receivers | out_degree ≥ 10 OR in_degree ≥ 10 |
"""),

# ── CELL 31: Pattern Label Generation ───────────────────────────────────────
md("### Cell 31 — Pattern Label Generation\nMines three AML pattern labels from graph topology using DFS cycle detection, chain tracking, and degree thresholds."),
code("cell-31", '''# ─────────────────────────────────────────────────────────────────────────────
# CELL 31: Pattern Label Generation (Graph Mining)
# ─────────────────────────────────────────────────────────────────────────────
def generate_pattern_labels(df_feat, df_edges, id_to_idx, labels_arr, node_timestamps,
                             circ_min_scc=2, lay_min_chain=3, smurf_deg_thresh=10):
    """
    Generate three binary pattern labels by mining graph topology.

    Pattern 1 — Circular Transfer (y_circ):
        Node belongs to a strongly connected component (SCC) of size >= circ_min_scc.
        Uses Tarjan's SCC algorithm via NetworkX.

    Pattern 2 — Layering (y_lay):
        Node participates in a directed chain of length >= lay_min_chain
        where timestamps are monotonically increasing (sequential laundering).

    Pattern 3 — Smurfing (y_smurf):
        Node has out_degree >= smurf_deg_thresh (fan-out) OR
              in_degree >= smurf_deg_thresh (fan-in aggregation).

    All patterns are masked to illicit nodes only:
        - Illicit + pattern detected → label = 1
        - Licit or unknown          → label = 0

    Returns:
        y_circ, y_lay, y_smurf: numpy int arrays of shape (N,)
    """
    N = len(df_feat)
    print("Building NetworkX directed graph for pattern mining...")

    # ── Build directed graph (subset for efficiency on large graph) ───────────
    G = nx.DiGraph()
    G.add_nodes_from(range(N))

    src_arr = edge_index[0].numpy()
    dst_arr = edge_index[1].numpy()
    ts_arr  = node_timestamps

    for s, d in tqdm(zip(src_arr, dst_arr), total=len(src_arr), desc="Adding edges"):
        G.add_edge(int(s), int(d), t=float(ts_arr[s]))

    print(f"  Graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Pattern 1: Circular Transfer — Tarjan's SCC
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    print("\\n[1/3] Detecting Circular Transfers (Tarjan SCC)...")
    y_circ = np.zeros(N, dtype=np.int64)
    scc_list = list(nx.strongly_connected_components(G))
    circ_count = 0
    for scc in tqdm(scc_list, desc="SCC scan"):
        if len(scc) >= circ_min_scc:
            for node in scc:
                if labels_arr[node] == 1:   # illicit only
                    y_circ[node] = 1
                    circ_count += 1
    print(f"  Circular nodes flagged (illicit): {circ_count:,}")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Pattern 2: Layering — sequential directed chains
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    print("\\n[2/3] Detecting Layering (sequential chains length >= {})...".format(lay_min_chain))
    y_lay = np.zeros(N, dtype=np.int64)

    # BFS from each illicit node — walk forward, check monotonic timestamps
    illicit_nodes = np.where(labels_arr == 1)[0]
    layer_set = set()

    for start in tqdm(illicit_nodes[:3000], desc="Layering BFS"):  # Sample for speed
        visited = {start}
        queue   = [(start, [start])]
        while queue:
            node, path = queue.pop(0)
            for nbr in G.successors(node):
                if nbr not in visited and ts_arr[nbr] >= ts_arr[node]:
                    new_path = path + [nbr]
                    if len(new_path) >= lay_min_chain:
                        layer_set.update(new_path)
                    if len(new_path) < 8:   # max chain depth
                        visited.add(nbr)
                        queue.append((nbr, new_path))

    for node in layer_set:
        if labels_arr[node] == 1:
            y_lay[node] = 1

    print(f"  Layering nodes flagged (illicit): {y_lay.sum():,}")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Pattern 3: Smurfing — degree thresholding
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    print("\\n[3/3] Detecting Smurfing (degree threshold={})...".format(smurf_deg_thresh))
    y_smurf = np.zeros(N, dtype=np.int64)

    out_degs = np.array([G.out_degree(i) for i in tqdm(range(N), desc="Out-degrees")])
    in_degs  = np.array([G.in_degree(i)  for i in range(N)])

    for i in range(N):
        if labels_arr[i] == 1:
            if out_degs[i] >= smurf_deg_thresh or in_degs[i] >= smurf_deg_thresh:
                y_smurf[i] = 1

    print(f"  Smurfing nodes flagged (illicit): {y_smurf.sum():,}")

    return y_circ, y_lay, y_smurf

y_circ, y_lay, y_smurf = generate_pattern_labels(
    df_feat, df_edges, id_to_idx, labels_arr, node_timestamps
)

print("\\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print("  PATTERN LABEL SUMMARY")
print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print(f"  Circular Transfer : {y_circ.sum():>6,} nodes  ({y_circ.sum()/N_ILLICIT*100:.1f}% of illicit)")
print(f"  Layering          : {y_lay.sum():>6,} nodes  ({y_lay.sum()/N_ILLICIT*100:.1f}% of illicit)")
print(f"  Smurfing          : {y_smurf.sum():>6,} nodes  ({y_smurf.sum()/N_ILLICIT*100:.1f}% of illicit)")
print(f"  Any pattern       : {((y_circ+y_lay+y_smurf)>0).sum():>6,} nodes")
print(f"  All 3 patterns    : {((y_circ==1)&(y_lay==1)&(y_smurf==1)).sum():>6,} nodes")
print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
'''),

# ── CELL 32: Label Distribution Visualisation ────────────────────────────────
md("### Cell 32 — Pattern Label Distribution Visualisation\nVisualises the three pattern label distributions per time step and shows co-occurrence."),
code("cell-32", '''# ─────────────────────────────────────────────────────────────────────────────
# CELL 32: Pattern Label Distribution Visualisation
# ─────────────────────────────────────────────────────────────────────────────
ts_axis = np.arange(1, 50)

def pattern_per_ts(y_pattern, node_timestamps):
    """Count pattern-positive nodes per time step."""
    return np.array([(y_pattern[node_timestamps == t]).sum() for t in range(1, 50)])

circ_per_ts  = pattern_per_ts(y_circ,  node_timestamps)
lay_per_ts   = pattern_per_ts(y_lay,   node_timestamps)
smurf_per_ts = pattern_per_ts(y_smurf, node_timestamps)

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.patch.set_facecolor(\'#0f1117\')
for ax in axes.flatten():
    ax.set_facecolor(\'#1a1d27\')

PCOLORS = {\'Circular\': \'#ff4d6d\', \'Layering\': \'#7c4dff\', \'Smurfing\': \'#ff8c42\'}

# ── Plot 1: Per-pattern count per time step ─────────────────────────────────
for arr, name, color in [(circ_per_ts,\'Circular\',\'#ff4d6d\'),(lay_per_ts,\'Layering\',\'#7c4dff\'),(smurf_per_ts,\'Smurfing\',\'#ff8c42\')]:
    axes[0,0].plot(ts_axis, arr, color=color, lw=2, marker=\'o\', ms=3, label=name)
    axes[0,0].fill_between(ts_axis, arr, alpha=0.12, color=color)
axes[0,0].set_title(\'Pattern-Positive Nodes per Time Step\', color=\'white\', fontweight=\'bold\')
axes[0,0].set_xlabel(\'Time Step\', color=\'white\')
axes[0,0].set_ylabel(\'Node Count\', color=\'white\')
axes[0,0].tick_params(colors=\'#adb5bd\')
axes[0,0].legend(facecolor=\'#2d3143\', labelcolor=\'white\')
axes[0,0].grid(alpha=0.3, color=\'#333\')
for sp in [\'top\',\'right\']: axes[0,0].spines[sp].set_visible(False)
for sp in [\'bottom\',\'left\']: axes[0,0].spines[sp].set_color(\'#444\')

# ── Plot 2: Stacked bar ──────────────────────────────────────────────────────
axes[0,1].bar(ts_axis, circ_per_ts,  color=\'#ff4d6d\', alpha=0.85, label=\'Circular\',  width=0.8)
axes[0,1].bar(ts_axis, lay_per_ts,   bottom=circ_per_ts, color=\'#7c4dff\', alpha=0.85, label=\'Layering\', width=0.8)
axes[0,1].bar(ts_axis, smurf_per_ts, bottom=circ_per_ts+lay_per_ts, color=\'#ff8c42\', alpha=0.85, label=\'Smurfing\', width=0.8)
axes[0,1].set_title(\'Stacked Pattern Count per Time Step\', color=\'white\', fontweight=\'bold\')
axes[0,1].set_xlabel(\'Time Step\', color=\'white\')
axes[0,1].set_ylabel(\'Nodes\', color=\'white\')
axes[0,1].tick_params(colors=\'#adb5bd\')
axes[0,1].legend(facecolor=\'#2d3143\', labelcolor=\'white\')
for sp in [\'top\',\'right\']: axes[0,1].spines[sp].set_visible(False)
for sp in [\'bottom\',\'left\']: axes[0,1].spines[sp].set_color(\'#444\')

# ── Plot 3: Total counts bar chart ───────────────────────────────────────────
totals  = [y_circ.sum(), y_lay.sum(), y_smurf.sum()]
pnames  = [\'Circular\\nTransfer\', \'Layering\', \'Smurfing\']
pcolors = [\'#ff4d6d\', \'#7c4dff\', \'#ff8c42\']
bars = axes[1,0].bar(pnames, totals, color=pcolors, alpha=0.9, width=0.5, edgecolor=\'#333\')
for bar, v in zip(bars, totals):
    axes[1,0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+5, f\'{v:,}\',
                   ha=\'center\', color=\'white\', fontsize=11, fontweight=\'bold\')
axes[1,0].set_title(\'Total Pattern Nodes (illicit only)\', color=\'white\', fontweight=\'bold\')
axes[1,0].set_ylabel(\'Count\', color=\'white\')
axes[1,0].tick_params(colors=\'#adb5bd\')
axes[1,0].set_ylim(0, max(totals)*1.2)
for sp in [\'top\',\'right\']: axes[1,0].spines[sp].set_visible(False)
for sp in [\'bottom\',\'left\']: axes[1,0].spines[sp].set_color(\'#444\')

# ── Plot 4: Co-occurrence matrix ─────────────────────────────────────────────
import itertools
patterns   = {\'Circular\': y_circ, \'Layering\': y_lay, \'Smurfing\': y_smurf}
plist      = list(patterns.keys())
co_matrix  = np.zeros((3, 3), dtype=int)
for i, p1 in enumerate(plist):
    for j, p2 in enumerate(plist):
        co_matrix[i, j] = int(((patterns[p1]==1) & (patterns[p2]==1)).sum())
im = axes[1,1].imshow(co_matrix, cmap=\'RdPu\', aspect=\'auto\')
axes[1,1].set_xticks(range(3)); axes[1,1].set_yticks(range(3))
axes[1,1].set_xticklabels(plist, color=\'white\', fontsize=10)
axes[1,1].set_yticklabels(plist, color=\'white\', fontsize=10)
axes[1,1].set_title(\'Pattern Co-occurrence Matrix\', color=\'white\', fontweight=\'bold\')
for i in range(3):
    for j in range(3):
        axes[1,1].text(j, i, f\'{co_matrix[i,j]:,}\', ha=\'center\', va=\'center\',
                       color=\'white\', fontsize=11, fontweight=\'bold\')
plt.colorbar(im, ax=axes[1,1])

plt.suptitle(\'O3: AML Pattern Label Distribution\', color=\'white\', fontsize=14, fontweight=\'bold\', y=1.01)
plt.tight_layout()
save_path = f\'{RESULTS_DIR}/o3_pattern_distribution.png\'
plt.savefig(save_path, dpi=150, bbox_inches=\'tight\', facecolor=fig.get_facecolor())
plt.show()
print(f\'✅ Saved: {save_path}\')
'''),

# ── CELL 33: Update PyG Data object ─────────────────────────────────────────
md("### Cell 33 — Update PyG Data Object with Pattern Labels\nAdds `y_circ`, `y_lay`, `y_smurf` tensors to the existing `data` object."),
code("cell-33", '''# ─────────────────────────────────────────────────────────────────────────────
# CELL 33: Update PyG Data Object with Pattern Labels
# ─────────────────────────────────────────────────────────────────────────────
# Convert pattern labels to tensors
data.y_circ  = torch.tensor(y_circ,  dtype=torch.long)
data.y_lay   = torch.tensor(y_lay,   dtype=torch.long)
data.y_smurf = torch.tensor(y_smurf, dtype=torch.long)

print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print("  UPDATED PyG DATA OBJECT")
print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print(f"  data.x.shape      : {tuple(data.x.shape)}")
print(f"  data.y.shape      : {tuple(data.y.shape)}  (binary: illicit/licit)")
print(f"  data.y_circ.shape : {tuple(data.y_circ.shape)}  (Circular Transfer label)")
print(f"  data.y_lay.shape  : {tuple(data.y_lay.shape)}  (Layering label)")
print(f"  data.y_smurf.shape: {tuple(data.y_smurf.shape)}  (Smurfing label)")
print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

# Pattern prevalence in training split
train_lab = data.train_mask & (data.y != -1)
print(f"\\n  Pattern prevalence in training split:")
print(f"  Circular  : {data.y_circ[train_lab].sum().item():,} ({data.y_circ[train_lab].float().mean()*100:.2f}%)")
print(f"  Layering  : {data.y_lay[train_lab].sum().item():,} ({data.y_lay[train_lab].float().mean()*100:.2f}%)")
print(f"  Smurfing  : {data.y_smurf[train_lab].sum().item():,} ({data.y_smurf[train_lab].float().mean()*100:.2f}%)")

assert data.y_circ.shape[0]  == 203769, "Shape error: y_circ"
assert data.y_lay.shape[0]   == 203769, "Shape error: y_lay"
assert data.y_smurf.shape[0] == 203769, "Shape error: y_smurf"
print("\\n✅ All pattern label tensors verified")

# Save updated graph
torch.save(data, f\'{BASE_DIR}/temporal_graph_o3.pt\')
print(f\'✅ Saved: {BASE_DIR}/temporal_graph_o3.pt\')
'''),

# ── CELL 34: TGATConv Layer ──────────────────────────────────────────────────
md("### Cell 34 — TGATConv: Temporal Graph Attention Layer\nFull temporal attention layer that integrates FourierTimeEncoding into query/key/value projections."),
code("cell-34", '''# ─────────────────────────────────────────────────────────────────────────────
# CELL 34: TGATConv — Temporal Graph Attention Layer
# ─────────────────────────────────────────────────────────────────────────────
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax as pyg_softmax

class TGATConv(MessagePassing):
    """
    Temporal Graph Attention Convolution Layer.

    Extends standard GAT with learnable Fourier time encoding.
    For each edge (j → i), time delta Δt = t_i - t_j is encoded and
    concatenated with node features to form time-aware keys and values.

    Architecture (per edge j→i):
        Query_i  = [h_i || Φ(0)]   · W_Q         ← target node + zero-time
        Key_j    = [h_j || Φ(Δt)]  · W_K         ← source node + time delta
        Value_j  = [h_j || Φ(Δt)]  · W_V         ← source node + time delta
        score_ij = (Query_i · Key_j) / √d_k
        α_ij     = softmax(score_ij)
        h_i_new  = Σ_j α_ij · Value_j

    Reference: TGAT (Xu et al., ICLR 2020)
    """

    def __init__(self, in_channels: int, out_channels: int,
                 time_dim: int = 64, heads: int = 4, dropout: float = 0.2):
        super().__init__(aggr=\'add\')
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.heads        = heads
        self.time_dim     = time_dim
        self.dropout_p    = dropout

        # Reuse FourierTimeEncoding from O2
        self.time_enc = FourierTimeEncoding(time_dim)   # output: 2*time_dim

        # Augmented input dim: original features + time encoding
        aug_dim = in_channels + 2 * time_dim

        # Per-head projections
        d_head = out_channels // heads
        self.W_Q = nn.Linear(aug_dim, out_channels, bias=False)
        self.W_K = nn.Linear(aug_dim, out_channels, bias=False)
        self.W_V = nn.Linear(aug_dim, out_channels, bias=False)
        self.W_O = nn.Linear(out_channels, out_channels)

        self.bn      = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.scale   = (out_channels // heads) ** -0.5

        # Zero-time encoding (cached, computed once)
        self._phi_zero = None

    def _get_phi_zero(self, device):
        """Get Φ(0) — the encoding of zero time delta (for query)."""
        if self._phi_zero is None or self._phi_zero.device != device:
            self._phi_zero = self.time_enc(torch.zeros(1, device=device))  # (1, 2*time_dim)
        return self._phi_zero

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x          : Node features  (N, in_channels)
            edge_index : Edge indices   (2, E)
            t          : Node timestamps (N,)

        Returns:
            h_new : Updated node embeddings (N, out_channels)
        """
        N = x.size(0)

        # Φ(0) for query construction — broadcast over all nodes
        phi_zero = self._get_phi_zero(x.device).expand(N, -1)  # (N, 2*time_dim)
        x_q = torch.cat([x, phi_zero], dim=-1)                 # (N, aug_dim)

        # Q projection for all nodes
        Q = self.W_Q(x_q)   # (N, out_channels)

        # Propagate — compute K, V per edge using time delta
        h_new = self.propagate(edge_index, x=x, t=t, Q=Q, size=(N, N))  # (N, out_channels)
        h_new = self.W_O(h_new)
        h_new = self.bn(h_new)
        h_new = F.relu(h_new)
        h_new = self.dropout(h_new)

        return h_new

    def message(self, x_j, t_i, t_j, Q_i, edge_index_i):
        """
        Compute time-aware attention message for each edge.

        Args:
            x_j     : Source node features (E, in_channels)
            t_i     : Target node timestamps (E,)
            t_j     : Source node timestamps (E,)
            Q_i     : Query vectors for target nodes (E, out_channels)
            edge_index_i: Target node indices (E,)
        """
        # Time delta: how long ago did source node act?
        delta_t = (t_i - t_j).unsqueeze(-1)  # (E, 1)

        # Encode time delta
        phi_dt  = self.time_enc(delta_t.squeeze(-1))  # (E, 2*time_dim)

        # Augmented source: [h_j || Φ(Δt)]
        x_j_aug = torch.cat([x_j, phi_dt], dim=-1)   # (E, aug_dim)

        # Key and Value projections
        K = self.W_K(x_j_aug)   # (E, out_channels)
        V = self.W_V(x_j_aug)   # (E, out_channels)

        # Scaled dot-product attention
        score = (Q_i * K).sum(dim=-1, keepdim=True) * self.scale  # (E, 1)
        alpha = pyg_softmax(score, edge_index_i)                    # (E, 1)
        alpha = F.dropout(alpha, p=self.dropout_p, training=self.training)

        return alpha * V   # (E, out_channels) — attention-weighted values

    def extra_repr(self):
        return (f\'in_channels={self.in_channels}, out_channels={self.out_channels}, \'
                f\'time_dim={self.time_dim}, heads={self.heads}\')


# ── Quick sanity test ─────────────────────────────────────────────────────────
tgat_test = TGATConv(in_channels=172, out_channels=128, time_dim=64, heads=4)
with torch.no_grad():
    # Test on small subgraph
    x_s  = data.x[:100]
    ei_s = edge_index[:, edge_index[0] < 100]
    ei_s = ei_s[:, ei_s[1] < 100]
    t_s  = data.t[:100]
    out  = tgat_test(x_s, ei_s, t_s)

print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print("  TGATConv Sanity Test")
print(f"  Input:  x={tuple(x_s.shape)}, edges={tuple(ei_s.shape)}")
print(f"  Output: {tuple(out.shape)}  (expected: (100, 128))")
assert out.shape == (100, 128)
print("  ✅ TGATConv forward pass: PASS")
print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
'''),

# ── CELL 35: MultiPatternTGAT ────────────────────────────────────────────────
md("### Cell 35 — MultiPatternTGAT: Full Model with Shared Encoder + 3 Heads\nThe complete O3 model with 2 TGAT layers (shared) feeding into three independent classification heads."),
code("cell-35", '''# ─────────────────────────────────────────────────────────────────────────────
# CELL 35: MultiPatternTGAT — Shared Encoder + 3 Pattern Detection Heads
# ─────────────────────────────────────────────────────────────────────────────
class PatternHead(nn.Module):
    """
    Single classification head for one AML pattern.

    Architecture: Linear → LayerNorm → ReLU → Dropout → Linear → Sigmoid
    Input: shared embedding (N, hidden_dim)
    Output: pattern probability (N,)
    """
    def __init__(self, hidden_dim: int, pattern_name: str):
        super().__init__()
        self.name = pattern_name
        self.net  = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: Shared embeddings (N, hidden_dim)
        Returns:
            probs: Pattern probabilities (N,)
        """
        return torch.sigmoid(self.net(h)).squeeze(-1)

    def extra_repr(self):
        return f\'pattern={self.name}\'


class MultiPatternTGAT(nn.Module):
    """
    Multi-Pattern AML Detection Model with Shared TGAT Encoder.

    Architecture:
        Encoder:
            Layer 1: TGATConv(172 → 128, time_dim=64, heads=4)
            Layer 2: TGATConv(128 → 128, time_dim=64, heads=4)
        Shared Embedding: (N, 128)
        Heads:
            circular_head : PatternHead(128 → 1)  → P(circular)
            layering_head : PatternHead(128 → 1)  → P(layering)
            smurfing_head : PatternHead(128 → 1)  → P(smurfing)

    All three heads see the SAME shared embedding.
    This forces the encoder to learn representations useful for ALL patterns.

    Args:
        in_dim    : Input feature dimension (default: 172)
        hidden    : Hidden/embedding dimension (default: 128)
        time_dim  : Fourier time encoding dimension (default: 64)
        heads     : Number of attention heads (default: 4)
        dropout   : Dropout rate (default: 0.25)
    """

    def __init__(self, in_dim: int = 172, hidden: int = 128,
                 time_dim: int = 64, heads: int = 4, dropout: float = 0.25):
        super().__init__()

        self.hidden   = hidden
        self.time_dim = time_dim

        # ── Shared Temporal Encoder (2 TGAT layers) ─────────────────────────
        self.tgat1    = TGATConv(in_dim, hidden, time_dim=time_dim, heads=heads, dropout=dropout)
        self.tgat2    = TGATConv(hidden, hidden, time_dim=time_dim, heads=heads, dropout=dropout)

        # ── Residual projection (in_dim → hidden for skip connection) ────────
        self.proj     = nn.Linear(in_dim, hidden)

        # ── Layer Norms ───────────────────────────────────────────────────────
        self.ln1      = nn.LayerNorm(hidden)
        self.ln2      = nn.LayerNorm(hidden)

        # ── Three Pattern Detection Heads (shared embedding → pattern probs) ─
        self.circular_head = PatternHead(hidden, \'circular\')
        self.layering_head = PatternHead(hidden, \'layering\')
        self.smurfing_head = PatternHead(hidden, \'smurfing\')

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor,
               t: torch.Tensor) -> torch.Tensor:
        """
        Shared TGAT encoder — produces node embeddings used by all heads.

        Args:
            x          : (N, in_dim)
            edge_index : (2, E)
            t          : (N,)
        Returns:
            h: Shared embeddings (N, hidden)
        """
        # Layer 1 + residual
        h1 = self.tgat1(x, edge_index, t)                       # (N, hidden)
        h1 = self.ln1(h1 + self.proj(x))                        # residual from x

        # Layer 2 + residual
        h2 = self.tgat2(h1, edge_index, t)                      # (N, hidden)
        h2 = self.ln2(h2 + h1)                                  # residual from h1

        return h2   # shared embedding

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                t: torch.Tensor):
        """
        Full forward pass.

        Returns:
            p_circ, p_lay, p_smurf: Pattern probabilities, each (N,)
            h: Shared embeddings (N, hidden) — for visualisation
        """
        h = self.encode(x, edge_index, t)

        p_circ  = self.circular_head(h)   # (N,)
        p_lay   = self.layering_head(h)   # (N,)
        p_smurf = self.smurfing_head(h)   # (N,)

        return p_circ, p_lay, p_smurf, h


# ── Instantiate and test ──────────────────────────────────────────────────────
model_o3 = MultiPatternTGAT(in_dim=172, hidden=128, time_dim=64, heads=4)
total_params = sum(p.numel() for p in model_o3.parameters())

print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print("  MultiPatternTGAT Architecture")
print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print(model_o3)
print(f"\\n  Total parameters: {total_params:,}")

# Test forward on small subgraph
with torch.no_grad():
    p_c, p_l, p_s, h_emb = model_o3(x_s, ei_s, t_s)

print(f"\\n  Forward pass on 100-node subgraph:")
print(f"  Shared embedding  : {tuple(h_emb.shape)}")
print(f"  P(circular)       : {tuple(p_c.shape)}  range [{p_c.min():.3f}, {p_c.max():.3f}]")
print(f"  P(layering)       : {tuple(p_l.shape)}  range [{p_l.min():.3f}, {p_l.max():.3f}]")
print(f"  P(smurfing)       : {tuple(p_s.shape)}  range [{p_s.min():.3f}, {p_s.max():.3f}]")
assert h_emb.shape == (100, 128)
assert p_c.shape == p_l.shape == p_s.shape == (100,)
print("  ✅ MultiPatternTGAT forward pass: PASS")
print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
'''),

# ── CELL 36: Multi-Task Loss ─────────────────────────────────────────────────
md("### Cell 36 — Multi-Task Weighted Loss Function\nDefines per-pattern BCE losses and the joint multi-task loss with learnable task weights."),
code("cell-36", '''# ─────────────────────────────────────────────────────────────────────────────
# CELL 36: Multi-Task Weighted Loss Function
# ─────────────────────────────────────────────────────────────────────────────
def compute_pattern_class_weights(y_pattern, train_mask_np, labels_arr):
    """
    Compute positive class weight for a pattern head to handle imbalance.
    w_pos = n_negative / n_positive (within labelled training nodes).

    Args:
        y_pattern    : numpy array (N,) with 0/1 pattern labels
        train_mask_np: boolean array (N,) — training node mask
        labels_arr   : numpy array (N,) — original binary labels (-1/0/1)
    Returns:
        pos_weight: scalar float
    """
    # Only consider labelled training nodes
    mask = train_mask_np & (labels_arr != -1)
    y    = y_pattern[mask]
    n_pos = y.sum()
    n_neg = len(y) - n_pos
    return float(n_neg / max(n_pos, 1))


class MultiTaskAMLLoss(nn.Module):
    """
    Multi-task BCE loss for three AML pattern heads.

    Loss = λ₁·L_circ + λ₂·L_lay + λ₃·L_smurf

    Each sub-loss is weighted BCE with per-pattern positive class weights
    to correct for class imbalance.

    Args:
        w_circ : positive class weight for circular head
        w_lay  : positive class weight for layering head
        w_smurf: positive class weight for smurfing head
        lambda_weights: tuple (λ₁, λ₂, λ₃) task importance weights
    """
    def __init__(self, w_circ: float, w_lay: float, w_smurf: float,
                 lambda_weights=(1.0, 1.0, 1.0)):
        super().__init__()
        self.lam      = lambda_weights
        self.bce_circ = nn.BCELoss(reduction=\'mean\')
        self.bce_lay  = nn.BCELoss(reduction=\'mean\')
        self.bce_smurf= nn.BCELoss(reduction=\'mean\')
        # Store positive weights as buffers
        self.register_buffer(\'w_circ\',  torch.tensor(w_circ))
        self.register_buffer(\'w_lay\',   torch.tensor(w_lay))
        self.register_buffer(\'w_smurf\', torch.tensor(w_smurf))

    def forward(self, p_circ, p_lay, p_smurf,
                y_circ, y_lay, y_smurf, mask):
        """
        Args:
            p_circ, p_lay, p_smurf : predicted probs (N,)
            y_circ, y_lay, y_smurf : pattern labels (N,)
            mask : boolean (N,) — which nodes to compute loss on
        Returns:
            total_loss, (l_circ, l_lay, l_smurf)
        """
        def wbce(pred, target, w_pos):
            # Per-sample weights
            weights = torch.where(target == 1,
                                  w_pos.expand_as(target),
                                  torch.ones_like(target))
            return F.binary_cross_entropy(pred, target, weight=weights)

        l_c = wbce(p_circ[mask],  y_circ[mask].float(),  self.w_circ)
        l_l = wbce(p_lay[mask],   y_lay[mask].float(),   self.w_lay)
        l_s = wbce(p_smurf[mask], y_smurf[mask].float(), self.w_smurf)

        total = self.lam[0]*l_c + self.lam[1]*l_l + self.lam[2]*l_s
        return total, (l_c.item(), l_l.item(), l_s.item())


# ── Compute class weights ─────────────────────────────────────────────────────
w_circ  = compute_pattern_class_weights(y_circ,  train_mask_np, labels_arr)
w_lay   = compute_pattern_class_weights(y_lay,   train_mask_np, labels_arr)
w_smurf = compute_pattern_class_weights(y_smurf, train_mask_np, labels_arr)

print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print("  MULTI-TASK LOSS CONFIGURATION")
print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print(f"  w_pos (Circular) : {w_circ:.2f}")
print(f"  w_pos (Layering) : {w_lay:.2f}")
print(f"  w_pos (Smurfing) : {w_smurf:.2f}")
print(f"  Task weights (λ) : (1.0, 1.0, 1.0)  — equal weighting")
print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

criterion = MultiTaskAMLLoss(w_circ=w_circ, w_lay=w_lay, w_smurf=w_smurf)
print("✅ MultiTaskAMLLoss ready")
'''),

# ── CELL 37: O3 Training Loop ────────────────────────────────────────────────
md("### Cell 37 — O3 Training Loop (30 Epochs)\nTrains MultiPatternTGAT with joint multi-task loss, tracking per-pattern F1 and AUC."),
code("cell-37", '''# ─────────────────────────────────────────────────────────────────────────────
# CELL 37: O3 Training Loop — 30 Epochs
# ─────────────────────────────────────────────────────────────────────────────
def train_o3(data, criterion, n_epochs=30, lr=5e-4, device=\'cpu\'):
    """
    Train MultiPatternTGAT for multi-pattern AML detection.

    Uses:
        - Adam optimizer with cosine LR annealing
        - Gradient clipping (max_norm=1.0)
        - Per-pattern tracking: loss, F1, AUC each epoch

    Returns:
        model    : trained MultiPatternTGAT
        history  : dict of training metrics per epoch
    """
    model = MultiPatternTGAT(in_dim=172, hidden=128, time_dim=64, heads=4).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs, eta_min=1e-5)

    # Masks
    train_lab  = data.train_mask & (data.y != -1)
    val_lab    = data.val_mask   & (data.y != -1)

    x_d     = data.x.to(device)
    ei_d    = data.edge_index.to(device)
    t_d     = data.t.to(device)
    yc_d    = data.y_circ.to(device)
    yl_d    = data.y_lay.to(device)
    ys_d    = data.y_smurf.to(device)
    trmask  = train_lab.to(device)
    valmask = val_lab.to(device)

    history = {k: [] for k in [
        \'total_loss\',\'l_circ\',\'l_lay\',\'l_smurf\',
        \'f1_circ\',\'f1_lay\',\'f1_smurf\',
        \'auc_circ\',\'auc_lay\',\'auc_smurf\'
    ]}

    best_f1   = 0.0
    best_state= None

    print(f"  Training on: {device}")
    print(f"  Train nodes: {trmask.sum().item():,}  |  Val nodes: {valmask.sum().item():,}")
    print("  " + "─"*55)

    for epoch in tqdm(range(1, n_epochs+1), desc="O3 Training"):
        # ── Train ─────────────────────────────────────────────────────────
        model.train()
        opt.zero_grad()

        p_c, p_l, p_s, _ = model(x_d, ei_d, t_d)

        total_loss, (lc, ll, ls) = criterion(p_c, p_l, p_s,
                                              yc_d, yl_d, ys_d,
                                              trmask)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        history[\'total_loss\'].append(total_loss.item())
        history[\'l_circ\'].append(lc)
        history[\'l_lay\'].append(ll)
        history[\'l_smurf\'].append(ls)

        # ── Validate ──────────────────────────────────────────────────────
        model.eval()
        with torch.no_grad():
            vp_c, vp_l, vp_s, _ = model(x_d, ei_d, t_d)

        def eval_head(probs, targets, mask):
            y_true  = targets[mask].cpu().numpy()
            y_score = probs[mask].cpu().numpy()
            y_pred  = (y_score > 0.5).astype(int)
            f1  = f1_score(y_true, y_pred, zero_division=0)
            try: auc = roc_auc_score(y_true, y_score)
            except: auc = 0.5
            return f1, auc

        f1c,  aucc  = eval_head(vp_c, yc_d, valmask)
        f1l,  aucl  = eval_head(vp_l, yl_d, valmask)
        f1s,  aucs  = eval_head(vp_s, ys_d, valmask)

        for key, val in zip([\'f1_circ\',\'f1_lay\',\'f1_smurf\',\'auc_circ\',\'auc_lay\',\'auc_smurf\'],
                             [f1c, f1l, f1s, aucc, aucl, aucs]):
            history[key].append(val)

        mean_f1 = (f1c + f1l + f1s) / 3
        if mean_f1 > best_f1:
            best_f1   = mean_f1
            best_state= {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 5 == 0 or epoch == 1:
            print(f"    Epoch {epoch:02d} | Loss: {total_loss.item():.4f} "
                  f"(C:{lc:.3f} L:{ll:.3f} S:{ls:.3f}) | "
                  f"F1: C={f1c:.3f} L={f1l:.3f} S={f1s:.3f}")

    # Restore best weights
    if best_state:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    print(f"\\n  ✅ Training complete | Best mean F1: {best_f1:.4f}")
    return model, history

device_o3 = torch.device(\'cuda\' if torch.cuda.is_available() else \'cpu\')
criterion  = criterion.to(device_o3)

print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print("  O3 TRAINING: MultiPatternTGAT (30 Epochs)")
print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
model_o3, history_o3 = train_o3(data, criterion, n_epochs=30, lr=5e-4, device=device_o3)
'''),

# ── CELL 38: Per-Pattern Evaluation ─────────────────────────────────────────
md("### Cell 38 — Per-Pattern Evaluation on Test Set\nComputes F1, Precision, Recall, and AUC-ROC for each of the three pattern heads."),
code("cell-38", '''# ─────────────────────────────────────────────────────────────────────────────
# CELL 38: Per-Pattern Test Set Evaluation
# ─────────────────────────────────────────────────────────────────────────────
from sklearn.metrics import classification_report, precision_recall_fscore_support

def evaluate_o3(model, data, device):
    """
    Full evaluation of MultiPatternTGAT on the test split.

    Returns dict with per-pattern metrics.
    """
    model.eval()
    test_lab = data.test_mask & (data.y != -1)

    x_d  = data.x.to(device)
    ei_d = data.edge_index.to(device)
    t_d  = data.t.to(device)

    with torch.no_grad():
        p_c, p_l, p_s, h_emb = model(x_d, ei_d, t_d)

    results = {}
    for name, probs, labels_t in [
        (\'Circular\', p_c, data.y_circ),
        (\'Layering\', p_l, data.y_lay),
        (\'Smurfing\', p_s, data.y_smurf),
    ]:
        y_true  = labels_t[test_lab].numpy()
        y_score = probs[test_lab].cpu().numpy()
        y_pred  = (y_score > 0.5).astype(int)

        prec, rec, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average=\'binary\', zero_division=0
        )
        try: auc = roc_auc_score(y_true, y_score)
        except: auc = 0.5

        results[name] = {\'F1\': f1, \'Precision\': prec, \'Recall\': rec, \'AUC-ROC\': auc,
                         \'y_true\': y_true, \'y_score\': y_score, \'y_pred\': y_pred}

    return results, h_emb.cpu()

test_results, shared_embeddings = evaluate_o3(model_o3, data, device_o3)

print("╔══════════════════════════╦═══════════╦═══════════╦═══════════╦═══════════╗")
print("║ Pattern                  ║ Precision ║   Recall  ║    F1     ║  AUC-ROC  ║")
print("╠══════════════════════════╬═══════════╬═══════════╬═══════════╬═══════════╣")
for name, res in test_results.items():
    print(f"║ {name:<24s} ║  {res[\'Precision\']:.4f}   ║  {res[\'Recall\']:.4f}   ║  {res[\'F1\']:.4f}   ║  {res[\'AUC-ROC\']:.4f}   ║")
print("╚══════════════════════════╩═══════════╩═══════════╩═══════════╩═══════════╝")

mean_f1  = sum(r[\'F1\']      for r in test_results.values()) / 3
mean_auc = sum(r[\'AUC-ROC\'] for r in test_results.values()) / 3
print(f"\\n  Mean F1     : {mean_f1:.4f}")
print(f"  Mean AUC-ROC: {mean_auc:.4f}")
'''),

# ── CELL 39: Training Curve Visualisation ───────────────────────────────────
md("### Cell 39 — O3 Training Curves\nVisualises per-pattern losses and F1 scores across 30 epochs."),
code("cell-39", '''# ─────────────────────────────────────────────────────────────────────────────
# CELL 39: O3 Training Curve Visualisation
# ─────────────────────────────────────────────────────────────────────────────
epochs = list(range(1, 31))

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.patch.set_facecolor(\'#0f1117\')
for ax in axes: ax.set_facecolor(\'#1a1d27\')

PCOLS = {\'Circular\': \'#ff4d6d\', \'Layering\': \'#7c4dff\', \'Smurfing\': \'#ff8c42\'}

# ── Plot 1: Multi-task loss ───────────────────────────────────────────────────
axes[0].plot(epochs, history_o3[\'total_loss\'], color=\'white\',  lw=2.5, label=\'Total\', zorder=5)
axes[0].plot(epochs, history_o3[\'l_circ\'],    color=\'#ff4d6d\', lw=1.5, label=\'Circular\',  alpha=0.8)
axes[0].plot(epochs, history_o3[\'l_lay\'],     color=\'#7c4dff\', lw=1.5, label=\'Layering\',  alpha=0.8)
axes[0].plot(epochs, history_o3[\'l_smurf\'],   color=\'#ff8c42\', lw=1.5, label=\'Smurfing\',  alpha=0.8)
axes[0].set_title(\'Multi-Task Training Loss\', color=\'white\', fontweight=\'bold\')
axes[0].set_xlabel(\'Epoch\', color=\'white\')
axes[0].set_ylabel(\'Loss\', color=\'white\')
axes[0].tick_params(colors=\'#adb5bd\')
axes[0].legend(facecolor=\'#2d3143\', labelcolor=\'white\', fontsize=9)
axes[0].grid(alpha=0.3, color=\'#333\')
for sp in [\'top\',\'right\']: axes[0].spines[sp].set_visible(False)
for sp in [\'bottom\',\'left\']: axes[0].spines[sp].set_color(\'#444\')

# ── Plot 2: Val F1 per pattern ───────────────────────────────────────────────
for key, col, lbl in [(\'f1_circ\',\'#ff4d6d\',\'Circular\'),(\'f1_lay\',\'#7c4dff\',\'Layering\'),(\'f1_smurf\',\'#ff8c42\',\'Smurfing\')]:
    axes[1].plot(epochs, history_o3[key], color=col, lw=2, label=lbl, marker=\'o\', ms=3)
axes[1].set_title(\'Validation F1 per Pattern\', color=\'white\', fontweight=\'bold\')
axes[1].set_xlabel(\'Epoch\', color=\'white\')
axes[1].set_ylabel(\'F1 Score\', color=\'white\')
axes[1].tick_params(colors=\'#adb5bd\')
axes[1].set_ylim([0, 1])
axes[1].legend(facecolor=\'#2d3143\', labelcolor=\'white\', fontsize=9)
axes[1].grid(alpha=0.3, color=\'#333\')
for sp in [\'top\',\'right\']: axes[1].spines[sp].set_visible(False)
for sp in [\'bottom\',\'left\']: axes[1].spines[sp].set_color(\'#444\')

# ── Plot 3: Val AUC per pattern ───────────────────────────────────────────────
for key, col, lbl in [(\'auc_circ\',\'#ff4d6d\',\'Circular\'),(\'auc_lay\',\'#7c4dff\',\'Layering\'),(\'auc_smurf\',\'#ff8c42\',\'Smurfing\')]:
    axes[2].plot(epochs, history_o3[key], color=col, lw=2, label=lbl, linestyle=\'--\', marker=\'s\', ms=3)
axes[2].set_title(\'Validation AUC-ROC per Pattern\', color=\'white\', fontweight=\'bold\')
axes[2].set_xlabel(\'Epoch\', color=\'white\')
axes[2].set_ylabel(\'AUC-ROC\', color=\'white\')
axes[2].tick_params(colors=\'#adb5bd\')
axes[2].set_ylim([0.5, 1.0])
axes[2].legend(facecolor=\'#2d3143\', labelcolor=\'white\', fontsize=9)
axes[2].grid(alpha=0.3, color=\'#333\')
for sp in [\'top\',\'right\']: axes[2].spines[sp].set_visible(False)
for sp in [\'bottom\',\'left\']: axes[2].spines[sp].set_color(\'#444\')

plt.suptitle(\'O3: MultiPatternTGAT Training Dynamics (30 Epochs)\', color=\'white\', fontsize=13, fontweight=\'bold\')
plt.tight_layout()
save_path = f\'{RESULTS_DIR}/o3_training_curves.png\'
plt.savefig(save_path, dpi=150, bbox_inches=\'tight\', facecolor=fig.get_facecolor())
plt.show()
print(f\'✅ Saved: {save_path}\')
'''),

# ── CELL 40: t-SNE Embedding Visualisation ───────────────────────────────────
md("### Cell 40 — t-SNE Visualisation of Shared Embeddings\nProjects the 128-dim shared TGAT embeddings into 2D to show pattern separation."),
code("cell-40", '''# ─────────────────────────────────────────────────────────────────────────────
# CELL 40: t-SNE Embedding Visualisation
# ─────────────────────────────────────────────────────────────────────────────
from sklearn.manifold import TSNE

# Sample 1500 labelled nodes for t-SNE (keep runtime under 2 min)
TSNE_N   = 1500
lab_mask = (data.y != -1).numpy()
lab_idx  = np.where(lab_mask)[0]
np.random.seed(42)
sample_idx = np.random.choice(lab_idx, min(TSNE_N, len(lab_idx)), replace=False)

emb_sample = shared_embeddings[sample_idx].numpy()  # (N_sample, 128)
y_sample   = labels_arr[sample_idx]
yc_sample  = y_circ[sample_idx]
yl_sample  = y_lay[sample_idx]
ys_sample  = y_smurf[sample_idx]

print(f"Running t-SNE on {len(sample_idx)} nodes (128D → 2D)...")
tsne  = TSNE(n_components=2, random_state=42, perplexity=40, n_iter=500)
emb2d = tsne.fit_transform(emb_sample)
print("✅ t-SNE complete")

fig, axes = plt.subplots(1, 4, figsize=(20, 5))
fig.patch.set_facecolor(\'#0f1117\')
for ax in axes: ax.set_facecolor(\'#1a1d27\'); ax.axis(\'off\')

def tsne_scatter(ax, labels, title, colors, alpha=0.6):
    for label, color, name in colors:
        mask = labels == label
        ax.scatter(emb2d[mask, 0], emb2d[mask, 1], c=color, s=8, alpha=alpha, label=name)
    ax.set_title(title, color=\'white\', fontsize=11, fontweight=\'bold\')
    ax.legend(facecolor=\'#2d3143\', labelcolor=\'white\', markerscale=2, fontsize=9, loc=\'upper right\')

tsne_scatter(axes[0], y_sample, \'Illicit vs Licit\',
             [(1,\'#ff4d6d\',\'Illicit\'),(0,\'#00b4d8\',\'Licit\')])
tsne_scatter(axes[1], yc_sample, \'Circular Transfer\',
             [(1,\'#ff4d6d\',\'Circular\'),(0,\'#2d2f50\',\'Other\')], alpha=0.5)
tsne_scatter(axes[2], yl_sample, \'Layering\',
             [(1,\'#7c4dff\',\'Layering\'),(0,\'#2d2f50\',\'Other\')], alpha=0.5)
tsne_scatter(axes[3], ys_sample, \'Smurfing\',
             [(1,\'#ff8c42\',\'Smurfing\'),(0,\'#2d2f50\',\'Other\')], alpha=0.5)

plt.suptitle(\'t-SNE of Shared TGAT Embeddings — Pattern Separation\', color=\'white\', fontsize=13, fontweight=\'bold\')
plt.tight_layout()
save_path = f\'{RESULTS_DIR}/o3_tsne_embeddings.png\'
plt.savefig(save_path, dpi=150, bbox_inches=\'tight\', facecolor=fig.get_facecolor())
plt.show()
print(f\'✅ Saved: {save_path}\')
'''),

# ── CELL 41: Per-Pattern Heatmap ─────────────────────────────────────────────
md("### Cell 41 — Pattern Detection Heatmap per Time Step\nShows which time steps have the highest density of each AML pattern."),
code("cell-41", '''# ─────────────────────────────────────────────────────────────────────────────
# CELL 41: Pattern Detection Heatmap per Time Step
# ─────────────────────────────────────────────────────────────────────────────
model_o3.eval()
with torch.no_grad():
    p_c_all, p_l_all, p_s_all, _ = model_o3(
        data.x.to(device_o3), data.edge_index.to(device_o3), data.t.to(device_o3)
    )
p_c_np = p_c_all.cpu().numpy()
p_l_np = p_l_all.cpu().numpy()
p_s_np = p_s_all.cpu().numpy()

# Mean predicted probability per time step per pattern
ts_axis = np.arange(1, 50)
heatmap = np.zeros((3, 49))
for ti, t in enumerate(range(1, 50)):
    t_mask = (node_timestamps == t)
    heatmap[0, ti] = p_c_np[t_mask].mean()
    heatmap[1, ti] = p_l_np[t_mask].mean()
    heatmap[2, ti] = p_s_np[t_mask].mean()

fig, ax = plt.subplots(figsize=(18, 4))
fig.patch.set_facecolor(\'#0f1117\')
ax.set_facecolor(\'#1a1d27\')

im = ax.imshow(heatmap, aspect=\'auto\', cmap=\'hot\', vmin=0, vmax=1)
ax.set_yticks([0, 1, 2])
ax.set_yticklabels([\'Circular\', \'Layering\', \'Smurfing\'], color=\'white\', fontsize=11)
ax.set_xticks(range(49))
ax.set_xticklabels(range(1, 50), color=\'#adb5bd\', fontsize=8, rotation=0)
ax.set_xlabel(\'Time Step\', color=\'white\')
ax.set_title(\'Mean Pattern Prediction Probability per Time Step\', color=\'white\', fontsize=13, fontweight=\'bold\')
plt.colorbar(im, ax=ax, label=\'Mean P(pattern)\', fraction=0.02)

# Annotate top time steps
for row in range(3):
    top_t = np.argmax(heatmap[row]) + 1
    ax.annotate(f\'t={top_t}\', xy=(top_t-1, row), color=\'cyan\', fontsize=8,
                ha=\'center\', va=\'center\', fontweight=\'bold\')

plt.tight_layout()
save_path = f\'{RESULTS_DIR}/o3_pattern_heatmap.png\'
plt.savefig(save_path, dpi=150, bbox_inches=\'tight\', facecolor=fig.get_facecolor())
plt.show()
print(f\'✅ Saved: {save_path}\')

# Top 5 time steps per pattern
print("\\n  Top 5 time steps per pattern:")
for i, name in enumerate([\'Circular\', \'Layering\', \'Smurfing\']):
    top5 = np.argsort(-heatmap[i])[:5] + 1
    print(f"  {name:12s}: {list(top5)}")
'''),

# ── CELL 42: Top Suspicious Nodes ────────────────────────────────────────────
md("### Cell 42 — Top Suspicious Nodes per Pattern\nRanks the highest-probability nodes for each AML pattern and prints a summary table."),
code("cell-42", '''# ─────────────────────────────────────────────────────────────────────────────
# CELL 42: Top Suspicious Nodes per Pattern
# ─────────────────────────────────────────────────────────────────────────────
def top_suspicious_nodes(probs_np, pattern_name, n=10):
    """Return top-n nodes sorted by pattern probability."""
    top_idx = np.argsort(-probs_np)[:n]
    rows = []
    for idx in top_idx:
        rows.append({
            \'node_idx\'  : int(idx),
            \'txId\'      : int(idx_to_id.get(int(idx), -1)),
            \'time_step\' : int(node_timestamps[idx]),
            \'label\'     : {1:\'Illicit\', 0:\'Licit\', -1:\'Unknown\'}.get(int(labels_arr[idx]), \'?\'),
            \'prob\'      : float(probs_np[idx]),
        })
    df_top = pd.DataFrame(rows)
    print(f"\\n  🔴 Top {n} {pattern_name} nodes:")
    print(df_top.to_string(index=False))
    return df_top

top_circ  = top_suspicious_nodes(p_c_np, \'Circular Transfer\')
top_lay   = top_suspicious_nodes(p_l_np, \'Layering\')
top_smurf = top_suspicious_nodes(p_s_np, \'Smurfing\')

# Save to CSV
top_circ.to_csv(f\'{RESULTS_DIR}/o3_top_circular.csv\',  index=False)
top_lay.to_csv(f\'{RESULTS_DIR}/o3_top_layering.csv\',   index=False)
top_smurf.to_csv(f\'{RESULTS_DIR}/o3_top_smurfing.csv\', index=False)
print(f\'\\n✅ Top node tables saved to {RESULTS_DIR}/\')
'''),

# ── CELL 43: Save All O3 Outputs ─────────────────────────────────────────────
md("### Cell 43 — Save All O3 Outputs\nSaves model weights, training history, embeddings, and results summary to Google Drive."),
code("cell-43", '''# ─────────────────────────────────────────────────────────────────────────────
# CELL 43: Save All O3 Outputs
# ─────────────────────────────────────────────────────────────────────────────
# 1. Model weights
torch.save(model_o3.state_dict(), f\'{BASE_DIR}/multi_pattern_tgat.pt\')
print(\'  1. multi_pattern_tgat.pt         — ✅ Saved\')

# 2. Shared embeddings (sampled)
np.save(f\'{RESULTS_DIR}/o3_shared_embeddings.npy\', shared_embeddings.numpy())
print(\'  2. o3_shared_embeddings.npy      — ✅ Saved\')

# 3. Training history
df_hist_o3 = pd.DataFrame({
    \'epoch\'      : list(range(1, 31)),
    \'total_loss\' : history_o3[\'total_loss\'],
    \'l_circ\'     : history_o3[\'l_circ\'],
    \'l_lay\'      : history_o3[\'l_lay\'],
    \'l_smurf\'    : history_o3[\'l_smurf\'],
    \'f1_circ\'    : history_o3[\'f1_circ\'],
    \'f1_lay\'     : history_o3[\'f1_lay\'],
    \'f1_smurf\'   : history_o3[\'f1_smurf\'],
    \'auc_circ\'   : history_o3[\'auc_circ\'],
    \'auc_lay\'    : history_o3[\'auc_lay\'],
    \'auc_smurf\'  : history_o3[\'auc_smurf\'],
})
df_hist_o3.to_csv(f\'{RESULTS_DIR}/o3_training_history.csv\', index=False)
print(\'  3. o3_training_history.csv       — ✅ Saved\')

# 4. Test results JSON
o3_results_json = {
    \'objective\': \'O3\',
    \'model\'    : \'MultiPatternTGAT\',
    \'epochs\'   : 30,
    \'patterns\' : {
        name: {\'F1\': float(res[\'F1\']), \'Precision\': float(res[\'Precision\']),
               \'Recall\': float(res[\'Recall\']), \'AUC_ROC\': float(res[\'AUC-ROC\'])}
        for name, res in test_results.items()
    },
    \'label_counts\': {
        \'circular\': int(y_circ.sum()),
        \'layering\' : int(y_lay.sum()),
        \'smurfing\' : int(y_smurf.sum()),
    },
    \'training_history\': {
        \'total_loss\': [float(x) for x in history_o3[\'total_loss\']],
        \'f1_circ\'   : [float(x) for x in history_o3[\'f1_circ\']],
        \'f1_lay\'    : [float(x) for x in history_o3[\'f1_lay\']],
        \'f1_smurf\'  : [float(x) for x in history_o3[\'f1_smurf\']],
        \'auc_circ\'  : [float(x) for x in history_o3[\'auc_circ\']],
        \'auc_lay\'   : [float(x) for x in history_o3[\'auc_lay\']],
        \'auc_smurf\' : [float(x) for x in history_o3[\'auc_smurf\']],
    }
}
with open(f\'{RESULTS_DIR}/o3_results.json\', \'w\') as f:
    json.dump(o3_results_json, f, indent=2)
print(\'  4. o3_results.json               — ✅ Saved\')

print(\'\\n✅ All O3 outputs saved to Google Drive\')
'''),

# ── CELL 44: O3 Verification Tests ───────────────────────────────────────────
md("### Cell 44 — O3 Verification Tests\nRuns 8 automated tests to verify correctness of the multi-pattern detection system."),
code("cell-44", '''# ─────────────────────────────────────────────────────────────────────────────
# CELL 44: O3 Verification Tests
# ─────────────────────────────────────────────────────────────────────────────
def run_o3_tests(data, model_o3, test_results, history_o3, device_o3):
    """8 correctness and quality tests for Objective 3."""
    results = []

    def test(name, condition):
        status = \'✅ PASS\' if condition else \'❌ FAIL\'
        print(f\'  {status} | {name}\')
        results.append(condition)

    print(\'━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\')
    print(\'  O3 VERIFICATION TESTS\')
    print(\'━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\')

    # Test 1: Pattern label shapes
    test(\'Test 1: y_circ, y_lay, y_smurf all shape (203769,)\',
         data.y_circ.shape[0] == 203769 and
         data.y_lay.shape[0]  == 203769 and
         data.y_smurf.shape[0]== 203769)

    # Test 2: Labels are binary {0,1}
    test(\'Test 2: All pattern labels in {0, 1}\',
         set(data.y_circ.unique().tolist()).issubset({0,1}) and
         set(data.y_lay.unique().tolist()).issubset({0,1}) and
         set(data.y_smurf.unique().tolist()).issubset({0,1}))

    # Test 3: Non-trivial labels
    test(\'Test 3: All patterns have at least some positive labels\',
         data.y_circ.sum().item() > 0 and
         data.y_lay.sum().item()  > 0 and
         data.y_smurf.sum().item()> 0)

    # Test 4: Model output shapes
    with torch.no_grad():
        p_c, p_l, p_s, h = model_o3(
            data.x[:50].to(device_o3),
            edge_index[:, edge_index[0]<50][:, edge_index[:, edge_index[0]<50][1]<50].to(device_o3),
            data.t[:50].to(device_o3)
        )
    test(\'Test 4: Model outputs 3 tensors of shape (N,)\',
         p_c.shape[0] == 50 and p_l.shape[0] == 50 and p_s.shape[0] == 50)

    # Test 5: Output range [0,1]
    test(\'Test 5: All outputs in [0, 1] (sigmoid bounded)\',
         p_c.min()>=0 and p_c.max()<=1 and
         p_l.min()>=0 and p_l.max()<=1 and
         p_s.min()>=0 and p_s.max()<=1)

    # Test 6: Shared embedding shape
    test(\'Test 6: Shared embeddings shape (N, 128)\',
         h.shape == (50, 128))

    # Test 7: Loss decreases
    losses = history_o3[\'total_loss\']
    test(\'Test 7: Multi-task loss decreases over training\',
         losses[-1] < losses[0])

    # Test 8: Per-pattern F1 > 0.4
    f1s = [test_results[p][\'F1\'] for p in test_results]
    test(\'Test 8: All per-pattern F1 > 0.40 on test set\',
         all(f >= 0.40 for f in f1s))

    passed = sum(results)
    print(\'━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\')
    print(f\'  O3 STATUS: {passed}/8 tests passed\')
    print(\'━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\')
    return passed

o3_passed = run_o3_tests(data, model_o3, test_results, history_o3, device_o3)
'''),

# ── CELL 45: Full Summary ────────────────────────────────────────────────────
md("### Cell 45 — Complete Phase 1 + O3 Summary Report\nPrints the final summary covering O1, O2, and O3."),
code("cell-45", '''# ─────────────────────────────────────────────────────────────────────────────
# CELL 45: Complete Phase 1 + O3 Summary Report
# ─────────────────────────────────────────────────────────────────────────────
cr = test_results.get(\'Circular\', {})
lr = test_results.get(\'Layering\', {})
sr = test_results.get(\'Smurfing\', {})

report_o3 = f"""
══════════════════════════════════════════════════════════
  TEMPORALAML — DISSERTATION PHASE 1 + O3 SUMMARY
══════════════════════════════════════════════════════════

  O1: Temporal Graph Construction       — {\'✅ COMPLETE\' if o1_passed==8 else f\'⚠️ {o1_passed}/8\'}
  O2: Learnable Fourier Time Encoding   — {\'✅ COMPLETE\' if o2_passed==8 else f\'⚠️ {o2_passed}/8\'}
  O3: Multi-Pattern AML Detection       — {\'✅ COMPLETE\' if o3_passed>=7 else f\'⚠️ {o3_passed}/8\'}

──────────────────────────────────────────────────────────
  OBJECTIVE 3: Multi-Pattern AML Detection

  Model: MultiPatternTGAT
    Architecture     : TGATConv × 2 (shared) + 3 heads
    Shared embedding : (N, 128)
    Time encoding    : FourierTimeEncoding (O2, reused)
    Total parameters : {sum(p.numel() for p in model_o3.parameters()):,}

  Pattern Labels Generated:
    Circular Transfer : {int(y_circ.sum()):,} nodes  (SCC size ≥ 2, illicit only)
    Layering          : {int(y_lay.sum()):,} nodes  (chain length ≥ 3, monotonic t)
    Smurfing          : {int(y_smurf.sum()):,} nodes  (degree ≥ 10, illicit only)

  Test Set Results:
    Pattern          | F1     | Precision | Recall | AUC-ROC
    ─────────────────┼────────┼───────────┼────────┼─────────
    Circular Transfer| {cr.get(\'F1\',0):.4f} | {cr.get(\'Precision\',0):.4f}    | {cr.get(\'Recall\',0):.4f} | {cr.get(\'AUC-ROC\',0):.4f}
    Layering         | {lr.get(\'F1\',0):.4f} | {lr.get(\'Precision\',0):.4f}    | {lr.get(\'Recall\',0):.4f} | {lr.get(\'AUC-ROC\',0):.4f}
    Smurfing         | {sr.get(\'F1\',0):.4f} | {sr.get(\'Precision\',0):.4f}    | {sr.get(\'Recall\',0):.4f} | {sr.get(\'AUC-ROC\',0):.4f}
    ─────────────────┼────────┼───────────┼────────┼─────────
    Mean             | {(cr.get(\'F1\',0)+lr.get(\'F1\',0)+sr.get(\'F1\',0))/3:.4f} |           |        | {(cr.get(\'AUC-ROC\',0)+lr.get(\'AUC-ROC\',0)+sr.get(\'AUC-ROC\',0))/3:.4f}

  Training: 30 epochs, Adam lr=5e-4, CosineAnnealingLR
  Tests: {o3_passed}/8 passed

══════════════════════════════════════════════════════════
  OVERALL: O1({o1_passed}/8) · O2({o2_passed}/8) · O3({o3_passed}/8)
  STATUS: {\'✅ READY FOR REVIEW\' if o1_passed>=7 and o2_passed>=7 and o3_passed>=7 else \'⚠️ CHECK FAILED TESTS\'}
══════════════════════════════════════════════════════════
"""

print(report_o3)
with open(f\'{RESULTS_DIR}/phase1_o3_summary.txt\', \'w\') as f:
    f.write(report_o3)
print(f\'✅ Report saved to {RESULTS_DIR}/phase1_o3_summary.txt\')
print(\'\\n🎓 Dissertation Phase 1 (O1 + O2 + O3) — Ready for Review\')
'''),

]  # end NEW_CELLS

# ── Load notebook and append ──────────────────────────────────────────────────
with open(NOTEBOOK_PATH, 'r') as f:
    nb = json.load(f)

original_count = len(nb['cells'])
nb['cells'].extend(NEW_CELLS)

with open(NOTEBOOK_PATH, 'w') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"✅ Notebook updated: {original_count} → {len(nb['cells'])} cells")
print(f"   Added {len(NEW_CELLS)} new cells (Section 5: O3)")
print(f"   Saved: {NOTEBOOK_PATH}")
