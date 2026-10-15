"""Tests of the GraphNeT model ports (weaver/nn/model/GraphNeT.py) that need no GraphNeT install.

The end-to-end comparison with GraphNeT itself and with the published NuBench predictions lives in
the NuBench project (utils/graphnet_reference.py); here: the building blocks against direct
reference implementations, and batch invariance / gradients of every backbone on packed batches.

    cd test && python -m unittest test_graphnet
"""

import os
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from weaver.nn.model.GraphNeT import (  # noqa: E402
    GraphNeTModel,
    PercentileClusters,
    counts_to_batch,
    knn_graph,
    sample_pulses,
    scatter_softmax,
    to_undirected,
)

DETECTOR = {
    "sensor_pos_x": ("expr", "(x / 100)"),
    "sensor_pos_y": ("expr", "(x / 100)"),
    "sensor_pos_z": ("expr", "(x / 1000)"),
    "t": ("expr", "(x / 1000000.0)"),
    "charge": ("expr", "torch.log10((1 + x))"),
}


def _pulses(counts, n_sensors=12, seed=0):
    """Packed pulses (x, y, z, t, charge) on a regular grid of sensors (many pulses per sensor)."""
    g = np.random.default_rng(seed)
    grid = np.stack(np.meshgrid(np.arange(3), np.arange(2), np.arange(2), indexing="ij"), -1).reshape(-1, 3)
    sensors = (grid[:n_sensors] * [50.0, 50.0, 30.0] - [40.0, 20.0, 2000.0]).astype(np.float32)
    rows = []
    for n in counts:
        s = sensors[g.integers(0, len(sensors), n)]
        t = g.uniform(0, 5000, n).astype(np.float32)
        q = g.exponential(1.0, n).astype(np.float32)
        rows.append(np.c_[s, t, q])
    values = torch.from_numpy(np.concatenate(rows).astype(np.float32))
    offsets = torch.zeros(len(counts) + 1, dtype=torch.long)
    offsets[1:] = torch.cumsum(torch.tensor(counts), 0)
    return values, offsets


class TestPercentileClusters(unittest.TestCase):
    def test_matches_numpy(self):
        values, offsets = _pulses([40, 7, 1, 25])
        counts = offsets[1:] - offsets[:-1]
        batch = counts_to_batch(counts, values.shape[0])
        nodes, node_batch = PercentileClusters()(values, batch)
        v = values.numpy()
        b = batch.numpy()
        ref = []
        for e in range(len(counts)):
            ve = v[b == e]
            # GraphNeT's lex_sort: by z, then y, then x
            keys = np.unique(ve[:, :3], axis=0)
            keys = keys[np.lexsort((keys[:, 0], keys[:, 1], keys[:, 2]))]
            for k in keys:
                sel = ve[np.all(ve[:, :3] == k, axis=1)]
                row = list(k.astype(np.float64))
                for f in (3, 4):
                    row += list(np.percentile(sel[:, f].astype(np.float64), [0, 10, 50, 90, 100]))
                row.append(np.log10(len(sel)))
                ref.append((e, row))
        np.testing.assert_array_equal(node_batch.numpy(), [e for e, _ in ref])
        np.testing.assert_array_equal(nodes.numpy(), np.array([r for _, r in ref]).astype(np.float32))


def _knn_reference(x, batch, k):
    """torch_cluster CUDA rule: k + 1 nearest including self (lower index first on ties), self dropped."""
    edges = set()
    for e in batch.unique():
        idx = torch.nonzero(batch == e).squeeze(1)
        for i in idx.tolist():
            d = ((x[idx] - x[i]) ** 2).sum(1)
            order = sorted(range(len(idx)), key=lambda j: (float(d[j]), j))[: k + 1]
            edges |= {(int(idx[j]), i) for j in order if int(idx[j]) != i}
    return edges


class TestGraphOps(unittest.TestCase):
    def test_knn_graph(self):
        g = torch.Generator().manual_seed(1)
        counts = torch.tensor([1, 5, 12, 9])
        # integer coordinates: lots of exact distance ties, including duplicate points
        x = torch.randint(0, 3, (int(counts.sum()), 3), generator=g).float()
        batch = counts_to_batch(counts, x.shape[0])
        ei = knn_graph(x, batch, counts, k=4, max_elements=64)  # small chunks
        self.assertEqual(set(map(tuple, ei.T.tolist())), _knn_reference(x, batch, 4))

    def test_to_undirected(self):
        ei = torch.tensor([[0, 1, 2, 2], [1, 0, 0, 3]])
        und = to_undirected(ei, 4)
        self.assertEqual(und.T.tolist(), [[0, 1], [0, 2], [1, 0], [2, 0], [2, 3], [3, 2]])

    def test_scatter_softmax(self):
        src = torch.randn(10, 2, 1)
        index = torch.tensor([0, 0, 1, 1, 1, 3, 3, 3, 3, 3])
        out = scatter_softmax(src, index, 4)
        for i in index.unique():
            torch.testing.assert_close(out[index == i], torch.softmax(src[index == i], 0), rtol=1e-6, atol=1e-6)

    def test_sample_pulses(self):
        counts = torch.tensor([3, 10, 8, 12])
        idx, new = sample_pulses(counts, 8)
        self.assertEqual(new.tolist(), [3, 8, 8, 8])
        batch = counts_to_batch(counts, int(counts.sum()))
        b = batch[idx]
        self.assertTrue(bool((b[1:] >= b[:-1]).all()))
        self.assertEqual(torch.bincount(b, minlength=4).tolist(), [3, 8, 8, 8])
        self.assertEqual(len(set(idx.tolist())), len(idx))


SPECS = {
    "DynEdge": dict(backbone="DynEdge", backbone_kwargs=dict(nb_inputs=14, global_pooling_schemes=["min", "max", "mean"])),
    "ParticleNeT": dict(backbone="ParticleNeT", backbone_kwargs=dict(nb_inputs=14, dropout_readout=0.0)),
    "ParticleNeT_pulses": dict(backbone="ParticleNeT", backbone_kwargs=dict(nb_inputs=5, dropout_readout=0.0),
                               node_definition="pulses", max_pulses=None),
    "GRIT": dict(
        backbone="GRIT",
        backbone_kwargs=dict(nb_inputs=14, hidden_dim=32, nb_outputs=3, n_layers=2, n_heads=4, attn_dropout=0.0,
                             activation="gelu", attn_activation="gelu", pred_head_activation="gelu"),
        task_kwargs=dict(disable_affine=True, hidden_size=3),
        edge_distance=True,
        input_feature_names=("sensor_pos_x", "sensor_pos_y", "sensor_pos_z", "t", "charge"),
    ),
    "DeepIce": dict(
        backbone="DeepIce",
        backbone_kwargs=dict(hidden_dim=64, seq_length=32, depth=2, head_size=16, depth_rel=1, n_features=5),
        node_definition="pulses",
        knn_k=None,
        max_pulses=20,
        sample_pulses=False,
    ),
}


class TestBackbones(unittest.TestCase):
    def _model(self, name):
        torch.manual_seed(0)
        kw = dict(SPECS[name])
        return GraphNeTModel(task="direction", detector=DETECTOR, **kw)

    def test_batch_invariance_and_grad(self):
        values, offsets = _pulses([30, 1, 8, 15, 22], seed=2)
        for name in SPECS:
            with self.subTest(model=name):
                model = self._model(name).eval()
                with torch.no_grad():
                    out = model((values, offsets))
                    single = torch.cat(
                        [
                            model((values[offsets[i] : offsets[i + 1]], offsets[i : i + 2] - offsets[i]))
                            for i in range(len(offsets) - 1)
                        ]
                    )
                self.assertEqual(out.shape, (5, 4))
                torch.testing.assert_close(single, out, rtol=1e-4, atol=1e-5)
                model.train()
                model((values, offsets))[:, :3].square().sum().backward()
                for pname, p in model.named_parameters():
                    if p.grad is None:
                        # GRIT's last layer updates the edges, which nothing downstream uses (as in GraphNeT)
                        last = f"backbone.layers.{len(getattr(model.backbone, 'layers', [])) - 1}."
                        self.assertTrue(pname.startswith(last) and "_e." in pname, f"{name}: no grad for {pname}")
                    else:
                        self.assertTrue(torch.isfinite(p.grad).all(), f"{name}: non-finite grad for {pname}")


if __name__ == "__main__":
    unittest.main()
