"""Pure-PyTorch ports of the GraphNeT models used by the NuBench benchmark (arXiv:2511.13111).

Ported from GraphNeT (https://github.com/graphnet-team/graphnet, ``src/graphnet/models``):

* ``DynEdge``      (``gnn/dynedge.py``)
* ``ParticleNeT``  (``gnn/particlenet.py``; GraphNeT's DGCNN variant, not weaver's ``ParticleNet``)
* ``GRIT``         (``gnn/grit.py`` + ``components/grit_layers.py``)
* ``DeepIce``      (``gnn/icemix.py`` + ``components/{embedding,attention_blocks}.py``)

together with the data representations NuBench trains them on: the ``PercentileClusters`` node
definition (one node per optical module, pulses summarized with percentiles), ``KNNEdges`` /
``KNNDistanceEdges`` and ``IceMixNodes`` (per-pulse sequence with random subsampling above
``max_pulses``), and the reconstruction ``Task`` heads.

No ``torch_geometric`` / ``torch_scatter`` / ``torch_cluster`` dependency: a batch is *packed*, i.e.
the per-pulse inputs of all events are concatenated, ``values (n_pulses_total, n_features)`` +
``offsets (n_events + 1,)`` (``weaver.utils.data.jagged.Unpadded``, delivered by a ``packed: true``
data-config input group under ``--jagged device``). Graph operations (k-NN, message passing,
pooling, sparse attention) work on the node/edge lists of the whole batch, as PyG does, so
nothing is padded except where the original model pads (DeepIce's dense transformer).

Module and parameter names follow GraphNeT's ``StandardModel`` (``backbone.*``,
``_tasks.0._affine.*``), so the published NuBench state dicts load with ``load_state_dict``.
The input standardization (GraphNeT ``Detector.feature_map``) is a constructor argument, since
the NuBench models use custom detector classes whose code only lives in the pickled models.
"""

import math
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

PULSE_FEATURES = ("sensor_pos_x", "sensor_pos_y", "sensor_pos_z", "t", "charge")

# GraphNeT detector standardizations used by the NuBench models (recovered from the pickled models):
# each entry is either ("div", d, s, m) -> m * (x / d - s) (computed in that order, so that the result is
# bitwise identical to the original expressions), or ("log10p1",) -> log10(1 + x).
DETECTORS = {
    # `MyDetector` of the DynEdge / ParticleNeT / DeepIce HexIceLE models
    "nubench": {
        "sensor_pos_x": ("div", 100.0, 0.0, 1.0),
        "sensor_pos_y": ("div", 100.0, 0.0, 1.0),
        "sensor_pos_z": ("div", 1000.0, 0.0, 1.0),
        "t": ("div", 1.0e6, 0.0, 1.0),
        "charge": ("log10p1",),
    },
    # `IceCubeWater` of the GRIT HexIceLE models: z / 500 + 4, 3 * (t / 1000 - 2.6)
    "icecube_water": {
        "sensor_pos_x": ("div", 500.0, 0.0, 1.0),
        "sensor_pos_y": ("div", 500.0, 0.0, 1.0),
        "sensor_pos_z": ("div", 500.0, -4.0, 1.0),
        "t": ("div", 1000.0, 2.6, 3.0),
        "charge": ("log10p1",),
    },
}


def _standardize(x: torch.Tensor, spec) -> torch.Tensor:
    if spec[0] == "log10p1":
        return torch.log10(1 + x)
    if spec[0] == "div":
        _, d, s, m = spec
        y = x / d
        if s != 0:
            y = y - s
        if m != 1:
            y = m * y
        return y
    if spec[0] == "identity":
        return x
    if spec[0] == "expr":
        # an expression of `x` recovered from the pickled GraphNeT detector (e.g. "torch.log10((1 + x))")
        return eval(spec[1], {"torch": torch, "__builtins__": {}}, {"x": x})
    raise ValueError(f"unknown standardization {spec}")


# ---------------------------------------------------------------------------------------------
# packed-batch helpers (stand-ins for torch_scatter / torch_cluster / torch_geometric.utils)
# ---------------------------------------------------------------------------------------------


def counts_to_batch(counts: torch.Tensor, total: int) -> torch.Tensor:
    return torch.repeat_interleave(torch.arange(len(counts), device=counts.device), counts, output_size=total)


def scatter_sum(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    return src.new_zeros((dim_size,) + src.shape[1:]).index_add_(0, index, src)


def scatter_mean(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    out = scatter_sum(src, index, dim_size)
    count = torch.bincount(index, minlength=dim_size).clamp_(min=1).to(src.dtype)
    return out / count.view((-1,) + (1,) * (src.dim() - 1))


def scatter_reduce(src: torch.Tensor, index: torch.Tensor, dim_size: int, reduce: str) -> torch.Tensor:
    """``amax`` / ``amin`` over rows; empty groups give 0 (like torch_scatter)."""
    idx = index.view((-1,) + (1,) * (src.dim() - 1)).expand_as(src)
    return src.new_zeros((dim_size,) + src.shape[1:]).scatter_reduce_(0, idx, src, reduce, include_self=False)


def scatter_softmax(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    """torch_geometric.utils.softmax over the groups of ``index`` (same arithmetic, incl. the 1e-16)."""
    src_max = scatter_reduce(src.detach(), index, dim_size, "amax")
    out = (src - src_max.index_select(0, index)).exp()
    out_sum = scatter_sum(out, index, dim_size) + 1e-16
    return out / out_sum.index_select(0, index)


def knn_graph(
    x: torch.Tensor,
    batch: torch.Tensor,
    counts: torch.Tensor,
    k: int,
    max_elements: int = 1 << 27,
) -> torch.Tensor:
    """``torch_cluster.knn_graph(x, k, batch, loop=False)``: edges ``(source j, target i)`` from each
    node's ``k`` nearest neighbours (squared Euclidean distance, self excluded) within its event.

    Distance ties are frequent (regular detector geometries; ReLU'd latent coordinates that are exactly
    0), so torch_cluster's CUDA kernel is emulated exactly: ``k + 1`` neighbours are searched
    *including* the node itself (which is then dropped, so a node with more than ``k``
    exact-zero-distance twins keeps ``k + 1`` neighbours), the squared distance is accumulated
    dimension by dimension with fused multiply-adds, and among equal distances the lower node index
    wins. torch_cluster's CPU path (a nanoflann KD-tree, used by GraphNeT's data loaders for the
    input graph) breaks ties differently.

    Implementation: the events are sorted by size and processed in chunks padded to the largest
    event of the chunk, with at most ``max_elements`` distance entries per chunk, so the cost follows
    sum(n_i^2) rather than n_events * max(n)^2. The selection is the ``(k + 1)``-th smallest distance
    ``t`` of each row (``topk``) plus a count: all entries below ``t`` and the lowest-index entries
    equal to ``t`` -- the first ``k + 1`` of a stable sort, without sorting.
    """
    n_nodes = x.shape[0]
    if n_nodes == 0:
        return torch.zeros(2, 0, dtype=torch.long, device=x.device)
    n_events = len(counts)
    offsets = torch.zeros(n_events + 1, dtype=torch.long, device=x.device)
    torch.cumsum(counts, 0, out=offsets[1:])
    counts_cpu = counts.tolist()
    order = sorted(range(n_events), key=counts_cpu.__getitem__)
    src_list, dst_list = [], []
    dim = max(1, x.shape[1])
    pos = 0
    while pos < n_events:
        # grow the chunk while (events * n_max^2 * dim) stays below the budget
        end = pos + 1
        while end < n_events and (end + 1 - pos) * counts_cpu[order[end]] ** 2 * dim <= max_elements:
            end += 1
        evs = order[pos:end]
        pos = end
        n_max = counts_cpu[evs[-1]]
        if n_max <= 1:
            continue
        kk = min(k + 1, n_max)
        ev_t = torch.tensor(evs, device=x.device)
        n_ev = counts[ev_t]
        j = torch.arange(n_max, device=x.device)
        valid = j.view(1, -1) < n_ev.view(-1, 1)  # (b, n)
        idx = (offsets[ev_t].view(-1, 1) + j.view(1, -1)).clamp(max=n_nodes - 1)
        d = x[idx]  # (b, n, dim)
        dist = None
        for c in range(d.shape[-1]):
            dc = d[..., c]
            diff = dc.unsqueeze(2) - dc.unsqueeze(1)  # (b, n, n): [i, j]
            if dist is None:
                dist = diff * diff
            else:  # fma(diff, diff, dist), as torch_cluster's kernel (addcmul is a fused multiply-add)
                dist = torch.addcmul(dist, diff, diff)
        dist = dist.masked_fill(~valid.unsqueeze(1), float("inf"))
        t = dist.topk(kk, dim=-1, largest=False).values[..., -1:]  # (b, n, 1)
        below = dist < t
        n_below = below.sum(-1, keepdim=True)
        equal = dist == t
        sel = below | (equal & (torch.cumsum(equal, -1) <= kk - n_below))
        sel &= torch.isfinite(dist) & valid.unsqueeze(-1)
        sel &= ~torch.eye(n_max, dtype=torch.bool, device=x.device)  # self
        b_i, i_i, j_i = sel.nonzero(as_tuple=True)
        base = offsets[ev_t][b_i]
        src_list.append(base + j_i)
        dst_list.append(base + i_i)
    if not src_list:
        return torch.zeros(2, 0, dtype=torch.long, device=x.device)
    return torch.stack([torch.cat(src_list), torch.cat(dst_list)], 0)


def to_undirected(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """``torch_geometric.utils.to_undirected`` (union with the reversed edges, coalesced: sorted by
    (row, col), duplicates removed)."""
    row, col = edge_index
    row, col = torch.cat([row, col]), torch.cat([col, row])
    key = torch.unique(row * num_nodes + col)
    return torch.stack([key // num_nodes, key % num_nodes], 0)


def homophily(edge_index: torch.Tensor, y: torch.Tensor, batch: torch.Tensor, dim_size: int) -> torch.Tensor:
    """``torch_geometric.utils.homophily(method='edge')`` per graph."""
    row, col = edge_index
    out = (y[row] == y[col]).to(y.dtype)
    return scatter_mean(out, batch[col], dim_size)


# ---------------------------------------------------------------------------------------------
# data representations
# ---------------------------------------------------------------------------------------------


def _numpy_linear_quantile(sorted_vals, starts, counts, q: float):
    """``np.percentile(..., method='linear')`` of each group of the (group-wise sorted) float64
    values, with numpy's exact arithmetic (``_lerp`` included)."""
    n = counts.to(torch.float64)
    virtual = (n - 1) * q
    prev = torch.floor(virtual)
    gamma = virtual - prev
    prev_i = prev.long()
    next_i = torch.minimum(prev_i + 1, counts - 1)
    a = sorted_vals[starts + prev_i]
    b = sorted_vals[starts + next_i]
    diff = b - a
    lerp = a + diff * gamma
    return torch.where(gamma >= 0.5, b - diff * (1 - gamma), lerp)


class PercentileClusters(nn.Module):
    """GraphNeT ``PercentileClusters(cluster_on=xyz, percentiles, add_counts=True)`` on a packed batch.

    One node per distinct (standardized) sensor position of an event, ordered like GraphNeT's
    ``lex_sort`` (by z, then y, then x). Node features: the position, the percentiles of every other
    input feature (in input order, computed in float64 with numpy's 'linear' method) and
    ``log10(number of pulses)``.
    """

    def __init__(self, percentiles: Sequence[int] = (0, 10, 50, 90, 100), add_counts: bool = True):
        super().__init__()
        self.percentiles = tuple(percentiles)
        self.add_counts = add_counts

    def forward(self, x: torch.Tensor, batch: torch.Tensor):
        """``x``: (pulses, 3 + F) standardized features, position first. Returns the node features
        (nodes, 3 + F * len(percentiles) + add_counts) float32 and the node -> event index."""
        pos = x[:, :3]
        order = torch.arange(x.shape[0], device=x.device)
        for key in (pos[:, 0], pos[:, 1], pos[:, 2], batch):  # least significant first
            order = order[torch.sort(key[order], stable=True).indices]
        p = pos[order]
        b = batch[order]
        new = torch.ones(x.shape[0], dtype=torch.bool, device=x.device)
        if x.shape[0] > 1:
            new[1:] = (b[1:] != b[:-1]) | (p[1:] != p[:-1]).any(dim=1)
        node = torch.cumsum(new.long(), 0) - 1
        n_nodes = int(node[-1]) + 1 if x.shape[0] else 0
        counts = torch.bincount(node, minlength=n_nodes)
        starts = torch.zeros_like(counts)
        starts[1:] = torch.cumsum(counts, 0)[:-1]

        columns = [p[new].to(torch.float64)]
        feats = x[order, 3:].to(torch.float64)
        for f in range(feats.shape[1]):
            v = feats[:, f]
            o = torch.sort(v, stable=True).indices
            o = o[torch.sort(node[o], stable=True).indices]
            sv = v[o]
            for pct in self.percentiles:
                columns.append(_numpy_linear_quantile(sv, starts, counts, pct / 100.0).unsqueeze(1))
        if self.add_counts:
            columns.append(torch.log10(counts.to(torch.float64)).unsqueeze(1))
        return torch.cat(columns, 1).to(torch.float32), b[new]


def sample_pulses(counts: torch.Tensor, max_pulses: int, random: bool = True):
    """``IceMixNodes`` pulse selection on a packed batch: events with ``n >= max_pulses`` keep a random
    subset of ``max_pulses`` pulses (in random order, as GraphNeT's ``randperm(n)[:max_pulses]``) or,
    with ``random=False``, their first ``max_pulses``. Returns the gather index and the new counts."""
    device = counts.device
    offsets = torch.zeros(len(counts) + 1, dtype=torch.long, device=device)
    torch.cumsum(counts, 0, out=offsets[1:])
    total = int(offsets[-1])
    batch = counts_to_batch(counts, total)
    long_ev = counts >= max_pulses
    if not bool(long_ev.any()):
        return torch.arange(total, device=device), counts
    pos = torch.arange(total, device=device) - offsets[batch]
    if random:
        # a random rank within each long event: sort (event, uniform key)
        key = torch.rand(total, device=device)
        o = torch.sort(key).indices
        o = o[torch.sort(batch[o], stable=True).indices]
        rank = torch.empty_like(pos)
        rank[o] = torch.arange(total, device=device) - offsets[batch[o]]
        keep_long = rank < max_pulses
        sel_order = torch.where(long_ev[batch], rank, pos)
    else:
        keep_long = pos < max_pulses
        sel_order = pos
    keep = torch.where(long_ev[batch], keep_long, torch.ones_like(keep_long))
    idx = torch.nonzero(keep).squeeze(1)
    # order the kept pulses of each event by their (random) rank
    idx = idx[torch.sort(sel_order[idx], stable=True).indices]
    idx = idx[torch.sort(batch[idx], stable=True).indices]
    return idx, torch.minimum(counts, torch.full_like(counts, max_pulses))


# ---------------------------------------------------------------------------------------------
# DynEdge / ParticleNeT
# ---------------------------------------------------------------------------------------------


class DynEdgeConv(nn.Module):
    """``EdgeConv(nn, aggr)`` followed by a k-NN regraphing on ``features_subset`` (GraphNeT)."""

    def __init__(self, mlp: nn.Module, aggr: str = "add", nb_neighbors: int = 8, features_subset=slice(0, 3)):
        super().__init__()
        self.nn = mlp
        self.aggr = aggr
        self.nb_neighbors = nb_neighbors
        self.features_subset = features_subset

    def forward(self, x, edge_index, batch, counts):
        src, dst = edge_index
        x_i = x.index_select(0, dst)
        x_j = x.index_select(0, src)
        msg = self.nn(torch.cat([x_i, x_j - x_i], dim=-1))
        if self.aggr == "add":
            x = scatter_sum(msg, dst, x.shape[0])
        elif self.aggr == "mean":
            x = scatter_mean(msg, dst, x.shape[0])
        elif self.aggr == "max":
            x = scatter_reduce(msg, dst, x.shape[0], "amax")
        else:
            raise ValueError(self.aggr)
        edge_index = knn_graph(x[:, self.features_subset], batch, counts, self.nb_neighbors)
        return x, edge_index


def _activation(name):
    if name is None or name.lower() == "relu":
        return nn.ReLU()
    if name.lower() == "gelu":
        return nn.GELU()
    raise ValueError(f"Activation layer {name} not supported.")


GLOBAL_POOLINGS = ("min", "max", "sum", "mean")


def _global_pool(x, batch, n_events, schemes):
    pooled = []
    for s in schemes:
        if s == "min":
            pooled.append(scatter_reduce(x, batch, n_events, "amin"))
        elif s == "max":
            pooled.append(scatter_reduce(x, batch, n_events, "amax"))
        elif s == "sum":
            pooled.append(scatter_sum(x, batch, n_events))
        elif s == "mean":
            pooled.append(scatter_mean(x, batch, n_events))
    return torch.cat(pooled, dim=1)


class DynEdge(nn.Module):
    """GraphNeT ``DynEdge`` (same constructor arguments and defaults)."""

    def __init__(
        self,
        nb_inputs: int,
        *,
        nb_neighbours: int = 8,
        features_subset=None,
        dynedge_layer_sizes=None,
        post_processing_layer_sizes=None,
        readout_layer_sizes=None,
        global_pooling_schemes=None,
        add_global_variables_after_pooling: bool = False,
        activation_layer: Optional[str] = None,
        add_norm_layer: bool = False,
        skip_readout: bool = False,
    ):
        super().__init__()
        self._features_subset = slice(0, 3) if features_subset is None else features_subset
        self._dynedge_layer_sizes = [tuple(s) for s in (dynedge_layer_sizes or [(128, 256), (336, 256), (336, 256), (336, 256)])]
        self._post_processing_layer_sizes = list(post_processing_layer_sizes or [336, 256])
        self._readout_layer_sizes = list(readout_layer_sizes or [128])
        if isinstance(global_pooling_schemes, str):
            global_pooling_schemes = [global_pooling_schemes]
        if global_pooling_schemes is not None:
            assert all(s in GLOBAL_POOLINGS for s in global_pooling_schemes)
        self._global_pooling_schemes = global_pooling_schemes
        if add_global_variables_after_pooling:
            assert self._global_pooling_schemes
        self._add_global_variables_after_pooling = add_global_variables_after_pooling
        self._activation = _activation(activation_layer)
        self._nb_inputs = nb_inputs
        self._nb_global_variables = 5 + nb_inputs
        self._nb_neighbours = nb_neighbours
        self._add_norm_layer = add_norm_layer
        self._skip_readout = skip_readout
        self.nb_outputs = self._readout_layer_sizes[-1]

        nb_input_features = nb_inputs
        if not self._add_global_variables_after_pooling:
            nb_input_features += self._nb_global_variables
        self._conv_layers = nn.ModuleList()
        nb_latent = nb_input_features
        for sizes in self._dynedge_layer_sizes:
            layers = []
            layer_sizes = [nb_latent] + list(sizes)
            for ix, (nb_in, nb_out) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
                if ix == 0:
                    nb_in *= 2
                layers.append(nn.Linear(nb_in, nb_out))
                if self._add_norm_layer:
                    layers.append(nn.LayerNorm(nb_out))
                layers.append(self._activation)
            self._conv_layers.append(
                DynEdgeConv(nn.Sequential(*layers), aggr="add", nb_neighbors=nb_neighbours, features_subset=self._features_subset)
            )
            nb_latent = nb_out
        nb_latent = sum(sizes[-1] for sizes in self._dynedge_layer_sizes) + nb_input_features
        layers = []
        layer_sizes = [nb_latent] + self._post_processing_layer_sizes
        for nb_in, nb_out in zip(layer_sizes[:-1], layer_sizes[1:]):
            layers.append(nn.Linear(nb_in, nb_out))
            if self._add_norm_layer:
                layers.append(nn.LayerNorm(nb_out))
            layers.append(self._activation)
        self._post_processing = nn.Sequential(*layers)
        nb_poolings = len(self._global_pooling_schemes) if self._global_pooling_schemes else 1
        nb_latent = nb_out * nb_poolings
        if self._add_global_variables_after_pooling:
            nb_latent += self._nb_global_variables
        layers = []
        layer_sizes = [nb_latent] + self._readout_layer_sizes
        for nb_in, nb_out in zip(layer_sizes[:-1], layer_sizes[1:]):
            layers.append(nn.Linear(nb_in, nb_out))
            layers.append(self._activation)
        self._readout = nn.Sequential(*layers)

    def forward(self, g: Dict[str, torch.Tensor]) -> torch.Tensor:
        x, edge_index, batch, counts, n_events = g["x"], g["edge_index"], g["batch"], g["counts"], g["n_events"]
        h = [homophily(edge_index, x[:, i], batch, n_events).view(-1, 1) for i in range(4)]
        global_variables = torch.cat(
            [scatter_mean(x, batch, n_events)] + h + [torch.log10(g["n_pulses"]).unsqueeze(1).to(x.dtype)], dim=1
        )
        if not self._add_global_variables_after_pooling:
            x = torch.cat([x, global_variables.index_select(0, batch)], dim=1)
        skip = [x]
        for conv in self._conv_layers:
            x, edge_index = conv(x, edge_index, batch, counts)
            skip.append(x)
        x = self._post_processing(torch.cat(skip, dim=1))
        if not self._skip_readout:
            if self._global_pooling_schemes:
                x = _global_pool(x, batch, n_events, self._global_pooling_schemes)
                if self._add_global_variables_after_pooling:
                    x = torch.cat([x, global_variables], dim=1)
            x = self._readout(x)
        return x


class ParticleNeT(nn.Module):
    """GraphNeT ``ParticleNeT`` (DGCNN with BatchNorm'd edge MLPs, mean aggregation)."""

    def __init__(
        self,
        nb_inputs: int,
        *,
        nb_neighbours: int = 16,
        features_subset=None,
        dynamic: bool = True,
        dynedge_layer_sizes=((64, 64, 64), (128, 128, 128), (256, 256, 256)),
        readout_layer_sizes=(256,),
        global_pooling_schemes="mean",
        activation_layer: Optional[str] = "relu",
        add_batchnorm_layer: bool = True,
        dropout_readout: float = 0.1,
        skip_readout: bool = False,
    ):
        super().__init__()
        self._features_subset = slice(0, 3) if features_subset is None else features_subset
        self._dynedge_layer_sizes = [tuple(s) for s in dynedge_layer_sizes]
        self._readout_layer_sizes = list(readout_layer_sizes or [256])
        if isinstance(global_pooling_schemes, str):
            global_pooling_schemes = [global_pooling_schemes]
        self._global_pooling_schemes = global_pooling_schemes
        self._activation = _activation(activation_layer)
        self._dynamic = dynamic
        self._skip_readout = skip_readout
        self.nb_outputs = self._readout_layer_sizes[-1]

        self._conv_layers = nn.ModuleList()
        nb_latent = nb_inputs
        for sizes in self._dynedge_layer_sizes:
            layers = []
            layer_sizes = [nb_latent] + list(sizes)
            for ix, (nb_in, nb_out) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
                if ix == 0:
                    nb_in *= 2
                layers.append(nn.Linear(nb_in, nb_out))
                if add_batchnorm_layer:
                    layers.append(nn.BatchNorm1d(nb_out))
                layers.append(self._activation)
            self._conv_layers.append(
                DynEdgeConv(nn.Sequential(*layers), aggr="mean", nb_neighbors=nb_neighbours, features_subset=self._features_subset)
            )
            nb_latent = nb_out
        nb_poolings = len(self._global_pooling_schemes) if self._global_pooling_schemes else 1
        nb_latent = nb_out * nb_poolings
        layers = []
        layer_sizes = [nb_latent] + self._readout_layer_sizes
        for nb_in, nb_out in zip(layer_sizes[:-1], layer_sizes[1:]):
            layers.append(nn.Linear(nb_in, nb_out))
            layers.append(self._activation)
            layers.append(nn.Dropout(dropout_readout))
        self._readout = nn.Sequential(*layers)

    def forward(self, g):
        x, edge_index, batch, counts, n_events = g["x"], g["edge_index"], g["batch"], g["counts"], g["n_events"]
        for conv in self._conv_layers:
            if self._dynamic:
                x, edge_index = conv(x, edge_index, batch, counts)
            else:
                x, _ = conv(x, edge_index, batch, counts)
        if not self._skip_readout:
            if self._global_pooling_schemes:
                x = _global_pool(x, batch, n_events, self._global_pooling_schemes)
            x = self._readout(x)
        return x


# ---------------------------------------------------------------------------------------------
# GRIT
# ---------------------------------------------------------------------------------------------


class LinearNodeEncoder(nn.Module):
    def __init__(self, dim_in, dim_emb):
        super().__init__()
        self.encoder = nn.Linear(dim_in, dim_emb)

    def forward(self, g):
        g["x"] = self.encoder(g["x"])
        return g


class LinearEdgeEncoder(nn.Module):
    def __init__(self, dim_emb):
        super().__init__()
        self.in_dim = 1
        self.encoder = nn.Linear(self.in_dim, dim_emb)

    def forward(self, g):
        g["edge_attr"] = self.encoder(g["edge_attr"].view(-1, self.in_dim))
        return g


class GritSparseMHA(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, use_bias, clamp=5.0, dropout=0.0, activation=nn.ReLU, edge_enhance=True):
        super().__init__()
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        self.clamp = abs(clamp) if clamp is not None else None
        self.edge_enhance = edge_enhance
        self.Q = nn.Linear(in_dim, out_dim * num_heads, bias=True)
        self.K = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)
        self.E = nn.Linear(in_dim, out_dim * num_heads * 2, bias=True)
        self.V = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)
        for lin in (self.Q, self.K, self.E, self.V):
            nn.init.xavier_normal_(lin.weight)
        self.Aw = nn.Parameter(torch.zeros(out_dim, num_heads, 1))
        nn.init.xavier_normal_(self.Aw)
        self.activation = activation()
        if self.edge_enhance:
            self.VeRow = nn.Parameter(torch.zeros(out_dim, num_heads, out_dim))
            nn.init.xavier_normal_(self.VeRow)

    def forward(self, g):
        x, edge_index = g["x"], g["edge_index"]
        n = x.shape[0]
        Q_x = self.Q(x).view(-1, self.num_heads, self.out_dim)
        K_x = self.K(x).view(-1, self.num_heads, self.out_dim)
        V_x = self.V(x).view(-1, self.num_heads, self.out_dim)
        E = self.E(g["edge_attr"]) if g.get("edge_attr") is not None else None
        score = K_x[edge_index[0]] + Q_x[edge_index[1]]
        if E is not None:
            E = E.view(-1, self.num_heads, self.out_dim * 2)
            E_w, E_b = E[:, :, : self.out_dim], E[:, :, self.out_dim :]
            score = score * E_w
            score = torch.sqrt(torch.relu(score)) - torch.sqrt(torch.relu(-score))
            score = score + E_b
        score = self.activation(score)
        e_t = score
        wE = score.flatten(1) if E is not None else None
        score = torch.einsum("ehd, dhc->ehc", score, self.Aw)
        if self.clamp is not None:
            score = torch.clamp(score, min=-self.clamp, max=self.clamp)
        score = scatter_softmax(score, edge_index[1], n).to(dtype=x.dtype)
        score = self.dropout(score)
        wV = scatter_sum(V_x[edge_index[0]] * score, edge_index[1], n)
        if self.edge_enhance and E is not None:
            rowV = scatter_sum(e_t * score, edge_index[1], n)
            wV = wV + torch.einsum("nhd, dhc -> nhc", rowV, self.VeRow)
        return wV, wE


class GritTransformerLayer(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        num_heads,
        dropout=0.0,
        norm=nn.BatchNorm1d,
        residual=True,
        deg_scaler=True,
        activation=nn.ReLU,
        norm_edges=True,
        update_edges=True,
        batch_norm_momentum=0.1,
        batch_norm_runner=True,
        rezero=False,
        enable_edge_transform=True,
        attn_bias=False,
        attn_dropout=0.0,
        attn_clamp=5.0,
        attn_activation=nn.ReLU,
        attn_edge_enhance=True,
    ):
        super().__init__()
        self.residual = residual
        self.update_edges = update_edges
        self.rezero = rezero
        self.deg_scaler = deg_scaler
        self.activation = activation()
        self.attention = GritSparseMHA(
            in_dim=in_dim,
            out_dim=out_dim // num_heads,
            num_heads=num_heads,
            use_bias=attn_bias,
            dropout=attn_dropout,
            clamp=attn_clamp,
            activation=attn_activation,
            edge_enhance=attn_edge_enhance,
        )
        hd = out_dim // num_heads * num_heads
        self.fc1_x = nn.Linear(hd, out_dim)
        self.fc1_e = nn.Linear(hd, out_dim) if enable_edge_transform else nn.Identity()
        if self.deg_scaler:
            self.deg_coef = nn.Parameter(torch.zeros(1, hd, 2))
            nn.init.xavier_normal_(self.deg_coef)

        def _norm():
            if norm is nn.LayerNorm:
                return nn.LayerNorm(out_dim)
            return nn.BatchNorm1d(out_dim, track_running_stats=batch_norm_runner, eps=1e-5, momentum=batch_norm_momentum)

        self.norm1_x = _norm()
        self.norm1_e = _norm() if norm_edges else nn.Identity()
        self.FFN_x_layer1 = nn.Linear(out_dim, out_dim * 2)
        self.FFN_x_layer2 = nn.Linear(out_dim * 2, out_dim)
        self.norm2_x = _norm()
        if self.rezero:
            self.alpha1_x = nn.Parameter(torch.zeros(1, 1))
            self.alpha2_x = nn.Parameter(torch.zeros(1, 1))
            self.alpha1_e = nn.Parameter(torch.zeros(1, 1))
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, g):
        x = g["x"]
        n = x.shape[0]
        deg = torch.bincount(g["edge_index"][0], minlength=n).to(x.dtype)
        log_deg = torch.log10(deg + 1).view(n, 1)
        x_attn_residual = x
        e_values_in = g.get("edge_attr")
        e = None
        x_attn_out, e_attn_out = self.attention(g)
        x = self.dropout1(x_attn_out.view(n, -1))
        if self.deg_scaler:
            x = torch.stack([x, x * log_deg], dim=-1)
            x = (x * self.deg_coef).sum(dim=-1)
        x = self.fc1_x(x)
        if e_attn_out is not None:
            e = self.fc1_e(self.dropout2(e_attn_out.flatten(1)))
        if self.residual:
            if self.rezero:
                x = x * self.alpha1_x
            x = x_attn_residual + x
            if e is not None:
                if self.rezero:
                    e = e * self.alpha1_e
                e = e + e_values_in
        x = self.norm1_x(x)
        if e is not None:
            e = self.norm1_e(e)
        x_ffn_residual = x
        x = self.FFN_x_layer2(self.dropout3(self.activation(self.FFN_x_layer1(x))))
        if self.residual:
            if self.rezero:
                x = x * self.alpha2_x
            x = x_ffn_residual + x
        x = self.norm2_x(x)
        g["x"] = x
        g["edge_attr"] = e if self.update_edges else e_values_in
        return g


class SANGraphHead(nn.Module):
    def __init__(self, dim_in, dim_out=1, L=2, activation=nn.ReLU, pooling="mean"):
        super().__init__()
        assert pooling in ("mean", "add")
        self.pooling = pooling
        fc_layers = [nn.Linear(dim_in // 2**n, dim_in // 2 ** (n + 1), bias=True) for n in range(L)]
        assert dim_in // 2**L >= dim_out, "Too much dim reduction!"
        fc_layers.append(nn.Linear(dim_in // 2**L, dim_out, bias=True))
        self.fc_layers = nn.ModuleList(fc_layers)
        self.L = L
        self.activation = activation()

    def forward(self, g):
        pool = scatter_mean if self.pooling == "mean" else scatter_sum
        h = pool(g["x"], g["batch"], g["n_events"])
        for i in range(self.L):
            h = self.activation(self.fc_layers[i](h))
        return self.fc_layers[self.L](h)


_ACTIVATIONS = {"relu": nn.ReLU, "gelu": nn.GELU, "silu": nn.SiLU}


def _act_cls(a):
    return _ACTIVATIONS[a.lower()] if isinstance(a, str) else a


class GRIT(nn.Module):
    """GraphNeT ``GRIT`` graph transformer (``position_encoding='NoPE'`` only)."""

    def __init__(
        self,
        nb_inputs: int,
        hidden_dim: int,
        nb_outputs: int = 1,
        ksteps: int = 21,
        n_layers: int = 10,
        n_heads: int = 8,
        pad_to_full_graph: bool = True,
        add_node_attr_as_self_loop: bool = False,
        dropout: float = 0.0,
        fill_value: float = 0.0,
        norm="batchnorm",
        attn_dropout: float = 0.2,
        edge_enhance: bool = True,
        update_edges: bool = True,
        attn_clamp: float = 5.0,
        activation="relu",
        attn_activation="relu",
        norm_edges: bool = True,
        enable_edge_transform: bool = True,
        pred_head_layers: int = 2,
        pred_head_activation="relu",
        pred_head_pooling: str = "mean",
        position_encoding: str = "NoPE",
    ):
        super().__init__()
        if position_encoding.lower() != "nope":
            raise NotImplementedError("only position_encoding='NoPE' is ported")
        norm_cls = nn.LayerNorm if (norm is nn.LayerNorm or str(norm).lower() == "layernorm") else nn.BatchNorm1d
        self.encoders = nn.ModuleList([LinearNodeEncoder(nb_inputs, hidden_dim), LinearEdgeEncoder(hidden_dim)])
        self.layers = nn.ModuleList(
            [
                GritTransformerLayer(
                    in_dim=hidden_dim,
                    out_dim=hidden_dim,
                    num_heads=n_heads,
                    dropout=dropout,
                    activation=_act_cls(activation),
                    attn_dropout=attn_dropout,
                    norm=norm_cls,
                    residual=True,
                    norm_edges=norm_edges,
                    enable_edge_transform=enable_edge_transform,
                    update_edges=update_edges,
                    attn_activation=_act_cls(attn_activation),
                    attn_clamp=attn_clamp,
                    attn_edge_enhance=edge_enhance,
                )
                for _ in range(n_layers)
            ]
        )
        self.head = SANGraphHead(
            dim_in=hidden_dim,
            dim_out=nb_outputs,
            L=pred_head_layers,
            activation=_act_cls(pred_head_activation),
            pooling=pred_head_pooling,
        )
        self.nb_outputs = nb_outputs

    def forward(self, g):
        g = dict(g)
        for enc in self.encoders:
            g = enc(g)
        for layer in self.layers:
            g = layer(g)
        return self.head(g)


# ---------------------------------------------------------------------------------------------
# DeepIce
# ---------------------------------------------------------------------------------------------


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim=16, n_freq=10000, scaled=False):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"dim has to be even. Got: {dim}")
        self.scale = nn.Parameter(torch.ones(1) * dim**-0.5) if scaled else 1.0
        self.dim = dim
        self.register_buffer("n_freq", torch.Tensor([n_freq]), persistent=False)

    def forward(self, x):
        half_dim = self.dim / 2
        emb = torch.log(self.n_freq) / half_dim
        emb = torch.exp(torch.arange(half_dim, device=x.device) * (-emb))
        emb = x.unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat((torch.sin(emb), torch.cos(emb)), dim=-1)
        return emb * self.scale


class FourierEncoder(nn.Module):
    def __init__(self, seq_length=128, mlp_dim=None, output_dim=384, scaled=False, n_features=6):
        super().__init__()
        self.sin_emb = SinusoidalPosEmb(dim=seq_length, scaled=scaled)
        self.sin_emb2 = SinusoidalPosEmb(dim=seq_length // 2, scaled=scaled)
        if n_features < 4:
            raise ValueError("At least x, y, z and time of the DOM are required.")
        elif n_features >= 6:
            self.aux_emb = nn.Embedding(2, seq_length // 2)
            hidden_dim = 6 * seq_length
        else:
            hidden_dim = int((n_features + 0.5) * seq_length)
        mlp_dim = mlp_dim or hidden_dim
        self.mlp = nn.Sequential(nn.Linear(hidden_dim, mlp_dim), nn.LayerNorm(mlp_dim), nn.GELU(), nn.Linear(mlp_dim, output_dim))
        self.n_features = n_features

    def forward(self, x, seq_length, max_len):
        length = torch.log10(seq_length.to(dtype=x.dtype))
        emb = [self.sin_emb(4096 * x[:, :, :3]).flatten(-2)]
        if self.n_features >= 5:
            emb.append(self.sin_emb(1024 * x[:, :, 4]))
        emb.append(self.sin_emb(4096 * x[:, :, 3]))
        if self.n_features >= 6:
            emb.append(self.aux_emb(x[:, :, 5].long()))
        emb.append(self.sin_emb2(length).unsqueeze(1).expand(-1, max_len, -1))
        return self.mlp(torch.cat(emb, -1))


class SpacetimeEncoder(nn.Module):
    def __init__(self, seq_length=32):
        super().__init__()
        self.sin_emb = SinusoidalPosEmb(dim=seq_length)
        self.projection = nn.Linear(seq_length, seq_length)

    def forward(self, x):
        pos = x[:, :, :3]
        time = x[:, :, 3]
        interval = (pos[:, :, None] - pos[:, None, :]).pow(2).sum(-1) - (
            (time[:, :, None] - time[:, None, :]) * (3e4 / 500 * 3e-1)
        ).pow(2)
        four_distance = torch.sign(interval) * torch.sqrt(torch.abs(interval))
        return self.projection(self.sin_emb(1024 * four_distance.clip(-4, 4)))


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, activation=nn.GELU, dropout_prob=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.input_projection = nn.Linear(in_features, hidden_features)
        self.activation = activation()
        self.output_projection = nn.Linear(hidden_features, out_features)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        return self.dropout(self.output_projection(self.activation(self.input_projection(x))))


class Attention_rel(nn.Module):
    def __init__(self, input_dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0, attn_head_dim=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = attn_head_dim or input_dim // num_heads
        all_head_dim = head_dim * num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.proj_q = nn.Linear(input_dim, all_head_dim, bias=False)
        self.proj_k = nn.Linear(input_dim, all_head_dim, bias=False)
        self.proj_v = nn.Linear(input_dim, all_head_dim, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, input_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q, k, v, rel_pos_bias=None, key_padding_mask=None):
        B, L, _ = q.shape
        q = F.linear(q, self.proj_q.weight, self.q_bias).reshape(B, L, self.num_heads, -1).permute(0, 2, 1, 3)
        k = F.linear(k, self.proj_k.weight, None).reshape(B, k.shape[1], self.num_heads, -1).permute(0, 2, 1, 3)
        v = F.linear(v, self.proj_v.weight, self.v_bias).reshape(B, v.shape[1], self.num_heads, -1).permute(0, 2, 1, 3)
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        if rel_pos_bias is not None:
            attn = attn + torch.einsum("bhic,bijc->bhij", q, rel_pos_bias)
        if key_padding_mask is not None:
            bias = torch.min(key_padding_mask[:, None, :], key_padding_mask[:, :, None])
            bias[torch.max(key_padding_mask[:, None, :], key_padding_mask[:, :, None]) < 0] = 0
            attn = attn + bias.unsqueeze(1)
        attn = self.attn_drop(attn.softmax(dim=-1))
        x = (attn @ v).transpose(1, 2)
        if rel_pos_bias is not None:
            x = x + torch.einsum("bhij,bijc->bihc", attn, rel_pos_bias)
        return self.proj_drop(self.proj(x.reshape(B, L, -1)))


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        random_tensor = x.new_empty((x.shape[0],) + (1,) * (x.ndim - 1)).bernoulli_(keep_prob)
        if keep_prob > 0.0:
            random_tensor.div_(keep_prob)
        return x * random_tensor


class Block_rel(nn.Module):
    def __init__(self, input_dim, num_heads, mlp_ratio=4.0, qkv_bias=False, qk_scale=None, dropout=0.0, attn_drop=0.0,
                 drop_path=0.0, init_values=None, activation=nn.GELU, norm_layer=nn.LayerNorm, attn_head_dim=None):
        super().__init__()
        self.norm1 = norm_layer(input_dim)
        self.attn = Attention_rel(input_dim, num_heads, attn_drop=attn_drop, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                  attn_head_dim=attn_head_dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(input_dim)
        self.mlp = Mlp(in_features=input_dim, hidden_features=int(input_dim * mlp_ratio), activation=activation,
                       dropout_prob=dropout)
        if init_values is not None:
            self.gamma_1 = nn.Parameter(init_values * torch.ones(input_dim))
            self.gamma_2 = nn.Parameter(init_values * torch.ones(input_dim))
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x, key_padding_mask=None, rel_pos_bias=None, kv=None):
        xn = self.norm1(x)
        kv = xn if kv is None else self.norm1(kv)
        a = self.attn(xn, kv, kv, rel_pos_bias=rel_pos_bias, key_padding_mask=key_padding_mask)
        if self.gamma_1 is None:
            x = x + self.drop_path(a)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.gamma_1 * self.drop_path(a))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class Block(nn.Module):
    def __init__(self, input_dim, num_heads, mlp_ratio=4.0, dropout=0.0, attn_drop=0.0, drop_path=0.0,
                 init_values=None, activation=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(input_dim)
        self.attn = nn.MultiheadAttention(input_dim, num_heads, dropout=attn_drop, batch_first=True)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(input_dim)
        self.mlp = Mlp(in_features=input_dim, hidden_features=int(input_dim * mlp_ratio), activation=activation,
                       dropout_prob=dropout)
        if init_values is not None:
            self.gamma_1 = nn.Parameter(init_values * torch.ones(input_dim))
            self.gamma_2 = nn.Parameter(init_values * torch.ones(input_dim))
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        xn = self.norm1(x)
        a = self.attn(xn, xn, xn, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=False)[0]
        if self.gamma_1 is None:
            x = x + self.drop_path(a)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.gamma_1 * a)
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class DeepIce(nn.Module):
    """GraphNeT ``DeepIce`` (IceCube Kaggle 1st place), without the optional DynEdge branch."""

    def __init__(self, hidden_dim=384, mlp_ratio=4, seq_length=192, depth=12, head_size=32, depth_rel=4, n_rel=1,
                 scaled_emb=False, include_dynedge=False, dynedge_args=None, n_features=6):
        super().__init__()
        if include_dynedge:
            raise NotImplementedError("DeepIce with include_dynedge=True is not ported")
        self.fourier_ext = FourierEncoder(seq_length=seq_length, mlp_dim=None, output_dim=hidden_dim, scaled=scaled_emb,
                                          n_features=n_features)
        self.rel_pos = SpacetimeEncoder(head_size)
        self.sandwich = nn.ModuleList([Block_rel(input_dim=hidden_dim, num_heads=hidden_dim // head_size) for _ in range(depth_rel)])
        self.cls_token = nn.Linear(hidden_dim, 1, bias=False)
        self.blocks = nn.ModuleList(
            [Block(input_dim=hidden_dim, num_heads=hidden_dim // head_size, mlp_ratio=mlp_ratio, drop_path=0.0, init_values=1)
             for _ in range(depth)]
        )
        self.n_rel = n_rel
        self.nb_outputs = hidden_dim

    def forward(self, g):
        x, batch, counts = g["x"], g["batch"], g["counts"]
        B = len(counts)
        L = int(counts.max())
        offsets = torch.zeros(B + 1, dtype=torch.long, device=x.device)
        torch.cumsum(counts, 0, out=offsets[1:])
        pos = torch.arange(x.shape[0], device=x.device) - offsets[batch]
        x0 = x.new_zeros(B, L, x.shape[1])
        x0[batch, pos] = x
        mask = torch.zeros(B, L, dtype=torch.bool, device=x.device)
        mask[batch, pos] = True
        h = self.fourier_ext(x0, counts, L)
        rel_pos_bias = self.rel_pos(x0)
        attn_mask = torch.zeros(mask.shape, device=x.device)
        attn_mask[~mask] = -torch.inf
        for i, blk in enumerate(self.sandwich):
            h = blk(h, attn_mask, rel_pos_bias)
            if i + 1 == self.n_rel:
                rel_pos_bias = None
        mask = torch.cat([torch.ones(B, 1, dtype=mask.dtype, device=x.device), mask], 1)
        attn_mask = torch.zeros(mask.shape, device=x.device)
        attn_mask[~mask] = -torch.inf
        h = torch.cat([self.cls_token.weight.unsqueeze(0).expand(B, -1, -1), h], 1)
        for blk in self.blocks:
            h = blk(h, None, attn_mask)
        return h[:, 0]


# ---------------------------------------------------------------------------------------------
# tasks and the full model
# ---------------------------------------------------------------------------------------------

# GraphNeT task heads: number of inputs of `_affine`, and `_forward` (inference transforms are the identity
# for the NuBench configurations: GraphNeT only applies `transform_inference` when `transform_target` is set)
TASKS = {
    # (dir_x, dir_y, dir_z, kappa)
    "direction": 3,
    # energy in GeV: softplus(x, beta=0.05) + eps
    "energy": 1,
    # vertex in metres: 100 * x
    "position": 3,
    # (x, y, z) in metres (100 * x) + (sigma_x, sigma_y, sigma_z) = |x| (metres)
    "position_uncertainty": 6,
    # sigmoid probability
    "track": 1,
    # raw logit
    "track_logits": 1,
    # sigmoid
    "inelasticity": 1,
    # sigmoid: the released GRIT models (and their published predictions) predate GraphNeT's current
    # `VisibleInelasticityReconstruction._forward` = 0.5 * (tanh(2x) + 1)
    "visible_inelasticity": 1,
}


class GraphNeTTask(nn.Module):
    """GraphNeT ``StandardLearnedTask``: ``_affine`` (Linear, or Identity with ``disable_affine``) + ``_forward``."""

    def __init__(self, kind: str, hidden_size: int, disable_affine: bool = False, position_scale: float = 100.0,
                 sigma_transform: str = "abs"):
        super().__init__()
        assert kind in TASKS, kind
        self.kind = kind
        self.nb_inputs = TASKS[kind]
        self._affine = nn.Identity() if disable_affine else nn.Linear(hidden_size, self.nb_inputs)
        self.position_scale = position_scale
        self.sigma_transform = sigma_transform

    def forward(self, x):
        x = self._affine(x)
        if self.kind == "direction":
            kappa = torch.linalg.vector_norm(x, dim=1) + torch.finfo(x.dtype).eps
            return torch.stack((x[:, 0] / kappa, x[:, 1] / kappa, x[:, 2] / kappa, kappa), dim=1)
        if self.kind == "energy":
            return F.softplus(x, beta=0.05) + torch.finfo(x.dtype).eps
        if self.kind == "position":
            return x * self.position_scale
        if self.kind == "position_uncertainty":
            mu = x[:, :3] * self.position_scale
            s = x[:, 3:]
            # the released ParticleNeT models output the raw values as sigma (checked against the published
            # predictions: no event with a negative raw sigma among those reproduced, so `abs` == identity there)
            if self.sigma_transform == "abs":
                s = s.abs()
            elif self.sigma_transform == "softplus":
                s = F.softplus(s)
            elif self.sigma_transform == "exp":
                s = torch.exp(s)
            return torch.cat([mu, s], dim=1)
        if self.kind in ("track", "inelasticity", "visible_inelasticity"):
            return torch.sigmoid(x)
        return x


class GraphNeTModel(nn.Module):
    """A GraphNeT ``StandardModel`` (data representation + backbone + one task) on packed pulses.

    Args:
        backbone: ``"DynEdge"``, ``"ParticleNeT"``, ``"GRIT"`` or ``"DeepIce"``.
        backbone_kwargs: constructor arguments, as in the GraphNeT model config.
        task: one of ``TASKS``; ``task_kwargs`` go to ``GraphNeTTask``.
        input_feature_names: the order of the pulse features the model sees (GraphNeT
            ``input_feature_names``), a permutation of ``PULSE_FEATURES``.
        detector: a key of ``DETECTORS`` or a feature -> standardization dict.
        node_definition: ``"percentile"`` (``PercentileClusters`` on xyz) or ``"pulses"`` (one node per pulse).
        knn_k: k of the graph definition's ``KNNEdges`` (``None``: no edges).
        edge_distance: ``KNNDistanceEdges`` (undirected, distance as the edge attribute, for GRIT).
        max_pulses / sample_pulses: ``IceMixNodes`` subsampling (``max_pulses=None``: GraphNeT's
            ``NodesAsPulses``, every pulse is a node).
        pulse_input_order: the order of the features in the packed input (data config).
    """

    def __init__(
        self,
        backbone: str,
        backbone_kwargs: Optional[dict] = None,
        task: str = "direction",
        task_kwargs: Optional[dict] = None,
        input_feature_names: Sequence[str] = ("sensor_pos_x", "sensor_pos_y", "sensor_pos_z", "charge", "t"),
        detector="nubench",
        node_definition: str = "percentile",
        percentiles: Sequence[int] = (0, 10, 50, 90, 100),
        knn_k: Optional[int] = 8,
        edge_distance: bool = False,
        max_pulses: Optional[int] = 800,
        sample_pulses: bool = True,
        pulse_input_order: Sequence[str] = PULSE_FEATURES,
    ):
        super().__init__()
        backbone_kwargs = dict(backbone_kwargs or {})
        self.input_feature_names = list(input_feature_names)
        self.feature_index = [list(pulse_input_order).index(f) for f in self.input_feature_names]
        assert self.input_feature_names[:3] == ["sensor_pos_x", "sensor_pos_y", "sensor_pos_z"]
        self.detector = DETECTORS[detector] if isinstance(detector, str) else dict(detector)
        self.node_definition = node_definition
        self.knn_k = knn_k
        self.edge_distance = edge_distance
        self.max_pulses = max_pulses
        self.sample_pulses = sample_pulses
        if node_definition == "percentile":
            self.nodes = PercentileClusters(percentiles)
        elif node_definition != "pulses":
            raise ValueError(node_definition)
        cls = {"DynEdge": DynEdge, "ParticleNeT": ParticleNeT, "GRIT": GRIT, "DeepIce": DeepIce}[backbone]
        self.backbone = cls(**backbone_kwargs)
        task_kwargs = dict(task_kwargs or {})
        task_kwargs.setdefault("hidden_size", self.backbone.nb_outputs)
        self._tasks = nn.ModuleList([GraphNeTTask(task, **task_kwargs)])

    def represent(self, values: torch.Tensor, offsets: torch.Tensor, edge_index: Optional[torch.Tensor] = None):
        """GraphNeT ``GraphDefinition.forward`` (minus the perturbation, done by the data loader)
        on the packed batch: standardize, build the nodes, draw the edges. A given ``edge_index``
        replaces the input graph (cross-checks with GraphNeT, whose data loaders build it with
        torch_cluster's CPU k-NN, which breaks distance ties in its own way)."""
        counts = offsets[1:] - offsets[:-1]
        n_events = len(counts)
        x = values[:, self.feature_index]
        x = torch.stack([_standardize(x[:, i], self.detector[f]) for i, f in enumerate(self.input_feature_names)], 1)
        n_pulses = counts
        batch = counts_to_batch(counts, x.shape[0])
        if self.node_definition == "percentile":
            x, batch = self.nodes(x, batch)
            counts = torch.bincount(batch, minlength=n_events)
        else:
            if self.max_pulses is not None:  # IceMixNodes; NodesAsPulses keeps every pulse
                idx, counts = sample_pulses(counts, self.max_pulses, random=self.sample_pulses)
                x = x[idx]
                batch = batch[idx]
        g = dict(x=x, batch=batch, counts=counts, n_events=n_events, n_pulses=n_pulses)
        if self.knn_k is not None:
            given = edge_index is not None
            if not given:
                edge_index = knn_graph(x[:, :3], batch, counts, self.knn_k)
            if self.edge_distance and not given:
                # KNNDistanceEdges: events without any edge get self loops, then undirected + distance
                has_edge = torch.zeros(x.shape[0], dtype=torch.bool, device=x.device)
                has_edge[edge_index[1]] = True
                lonely = torch.nonzero(~has_edge & (counts[batch] == 1)).squeeze(1)
                edge_index = torch.cat([edge_index, lonely.unsqueeze(0).expand(2, -1)], 1)
                edge_index = to_undirected(edge_index, x.shape[0])
            if self.edge_distance:
                src, tgt = edge_index
                g["edge_attr"] = torch.norm(x[src, :3] - x[tgt, :3], p=2, dim=-1).unsqueeze(1)
            g["edge_index"] = edge_index
        return g

    def forward(self, pulses, edge_index: Optional[torch.Tensor] = None):
        values, offsets = pulses
        g = self.represent(values, offsets, edge_index)
        return self._tasks[0](self.backbone(g))

    def load_graphnet_state_dict(self, state_dict, strict: bool = True):
        """Load a GraphNeT ``StandardModel`` state dict (``backbone.*``, ``_tasks.0.*``)."""
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        keep = {k: v for k, v in state_dict.items() if k.startswith("backbone.") or k.startswith("_tasks.")}
        return self.load_state_dict(keep, strict=strict)
