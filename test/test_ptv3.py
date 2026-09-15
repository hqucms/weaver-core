"""Unit tests for the pure-PyTorch Point Transformer V3 port (weaver/nn/model/PointTransformerV3.py).

The components that replace spconv / torch_scatter / flash-attn are checked against
reference implementations: the submanifold conv against a dense ``F.conv3d`` (one point
per voxel) and a brute-force reference (several points per voxel); the vectorized patch
layout and the attention against the original PTv3 code (copied below); the serialization
against the original ``serialization`` package when the PTv3 checkout is available. The
event-level wrapper is checked for padding and batch-composition invariance.
"""

import os
import sys
import unittest

import torch
import torch.nn.functional as F

from weaver.nn.model.PointTransformerV3 import (
    Point,
    PointTransformerV3,
    PTv3EventRegressor,
    SerializedAttention,
    SubMConv3d,
    encode,
    hilbert_decode,
    hilbert_encode,
    offset2bincount,
    z_order_decode,
    z_order_encode,
)

_PTV3_REPO = os.environ.get("PTV3_REPO", "/data/hqu/dev/nubench/PointTransformerV3")

_SMALL = dict(
    order=("z", "z-trans", "hilbert", "hilbert-trans"),
    stride=(2, 2),
    enc_depths=(1, 1, 2),
    enc_channels=(8, 16, 32),
    enc_num_head=(1, 2, 4),
    enc_patch_size=(8, 8, 8),
    dec_depths=(1, 1),
    dec_channels=(8, 16),
    dec_num_head=(1, 2),
    dec_patch_size=(8, 8),
    drop_path=0.0,
)


def _original_padding_and_inverse(offset, patch_size):
    """Verbatim (loop) version of ``SerializedAttention.get_padding_and_inverse`` from PTv3."""
    bincount = offset2bincount(offset)
    bincount_pad = (torch.div(bincount + patch_size - 1, patch_size, rounding_mode="trunc") * patch_size)
    mask_pad = bincount > patch_size
    bincount_pad = ~mask_pad * bincount + mask_pad * bincount_pad
    _offset = F.pad(offset, (1, 0))
    _offset_pad = F.pad(torch.cumsum(bincount_pad, dim=0), (1, 0))
    pad = torch.arange(_offset_pad[-1], device=offset.device)
    unpad = torch.arange(_offset[-1], device=offset.device)
    cu_seqlens = []
    for i in range(len(offset)):
        unpad[_offset[i] : _offset[i + 1]] += _offset_pad[i] - _offset[i]
        if bincount[i] != bincount_pad[i]:
            pad[_offset_pad[i + 1] - patch_size + (bincount[i] % patch_size) : _offset_pad[i + 1]] = pad[
                _offset_pad[i + 1] - 2 * patch_size + (bincount[i] % patch_size) : _offset_pad[i + 1] - patch_size
            ]
        pad[_offset_pad[i] : _offset_pad[i + 1]] -= _offset_pad[i] - _offset[i]
        cu_seqlens.append(
            torch.arange(_offset_pad[i], _offset_pad[i + 1], step=patch_size, dtype=torch.int32, device=offset.device)
        )
    cu_seqlens = F.pad(torch.concat(cu_seqlens), (0, 1), value=_offset_pad[-1])
    return pad, unpad, cu_seqlens


def _make_point(counts, grid=12, channels=8, seed=0, unique_voxels=False):
    g = torch.Generator().manual_seed(seed)
    coords, batch = [], []
    for b, n in enumerate(counts):
        if unique_voxels:
            flat = torch.randperm(grid**3, generator=g)[:n]
            c = torch.stack([flat // grid**2, (flat // grid) % grid, flat % grid], dim=1)
        else:
            c = torch.randint(0, grid, (n, 3), generator=g)
        coords.append(c)
        batch.append(torch.full((n,), b, dtype=torch.long))
    grid_coord = torch.cat(coords).int()
    batch = torch.cat(batch)
    feat = torch.randn(len(batch), channels, generator=g)
    return Point(feat=feat, grid_coord=grid_coord, batch=batch)


class TestSerialization(unittest.TestCase):
    def test_roundtrip(self):
        g = torch.Generator().manual_seed(1)
        for depth in (4, 10, 16):
            xyz = torch.randint(0, 2**depth, (500, 3), generator=g)
            torch.testing.assert_close(z_order_decode(z_order_encode(xyz, depth), depth), xyz)
            torch.testing.assert_close(hilbert_decode(hilbert_encode(xyz, depth), depth), xyz)

    def test_single_point(self):
        xyz = torch.tensor([[3, 5, 7]])
        self.assertEqual(hilbert_encode(xyz, 8).shape, (1,))
        self.assertEqual(encode(xyz, torch.zeros(1, dtype=torch.long), 8, "hilbert").shape, (1,))

    @unittest.skipUnless(os.path.isdir(os.path.join(_PTV3_REPO, "serialization")), "PTv3 checkout not found")
    def test_matches_original(self):
        sys.path.insert(0, _PTV3_REPO)
        try:
            from serialization import encode as original_encode
        finally:
            sys.path.pop(0)
        g = torch.Generator().manual_seed(2)
        xyz = torch.randint(0, 2**12, (1000, 3), generator=g)
        batch = torch.randint(0, 5, (1000,), generator=g).sort().values
        for order in ("z", "z-trans", "hilbert", "hilbert-trans"):
            torch.testing.assert_close(encode(xyz, batch, 12, order), original_encode(xyz, batch, 12, order))


class TestSubMConv3d(unittest.TestCase):
    def _dense_reference(self, conv, point, num_batches, grid):
        k, r = conv.kernel_size, conv.kernel_size // 2
        dense = torch.zeros(num_batches, conv.in_channels, grid, grid, grid)
        x, y, z = point.grid_coord.long().unbind(1)
        dense[point.batch, :, x, y, z] = point.feat
        # (K^3, Cin, Cout) with offsets in (dx, dy, dz) "ij" order -> (Cout, Cin, k, k, k)
        weight = conv.weight.reshape(k, k, k, conv.in_channels, conv.out_channels).permute(4, 3, 0, 1, 2)
        out = F.conv3d(dense, weight, conv.bias, padding=r)
        return out[point.batch, :, x, y, z]

    def test_matches_dense_conv_unique_voxels(self):
        for k in (3, 5):
            point = _make_point([40, 25, 60], grid=10, channels=6, unique_voxels=True, seed=k)
            conv = SubMConv3d(6, 7, kernel_size=k, bias=True, indice_key="t")
            ref = self._dense_reference(conv, point, 3, 10)
            out = conv(Point(point)).feat
            torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)

    def test_duplicate_points_bruteforce(self):
        point = _make_point([30, 20], grid=4, channels=3, seed=3)
        conv = SubMConv3d(3, 5, kernel_size=3, bias=True)
        out = conv(Point(point)).feat
        coords, batch, feat = point.grid_coord.long(), point.batch, point.feat
        ref = torch.zeros_like(out)
        for i in range(len(batch)):
            acc = conv.bias.clone()
            for kidx in range(27):
                d = torch.tensor([kidx // 9 - 1, (kidx // 3) % 3 - 1, kidx % 3 - 1])
                if kidx == 13:
                    acc = acc + feat[i] @ conv.weight[kidx]
                    continue
                sel = (batch == batch[i]) & (coords == coords[i] + d).all(1)
                if sel.any():
                    acc = acc + feat[sel].mean(0) @ conv.weight[kidx]
            ref[i] = acc
        torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)


class TestSerializedAttention(unittest.TestCase):
    def test_patch_layout_matches_original(self):
        for K in (1, 4, 7, 16):
            for counts in ([5, 16, 17, 3, 32, 1], [4, 4, 4], [40, 2, 7]):
                offset = torch.cumsum(torch.tensor(counts), 0)
                point = Point(offset=offset, feat=torch.zeros(offset[-1], 1))
                attn = SerializedAttention(8, 2, patch_size=K)
                for got, ref in zip(attn.get_padding_and_inverse(point), _original_padding_and_inverse(offset, K)):
                    torch.testing.assert_close(got, ref.to(got.dtype), msg=f"K={K} counts={counts}")

    def _original_attention(self, attn, point, K):
        """Original non-flash math (patches of exactly K points)."""
        H, C = attn.num_heads, attn.channels
        pad, unpad, _ = _original_padding_and_inverse(point.offset, K)
        order = point.serialized_order[attn.order_index][pad]
        inverse = unpad[point.serialized_inverse[attn.order_index]]
        qkv = attn.qkv(point.feat)[order]
        q, k, v = qkv.reshape(-1, K, 3, H, C // H).permute(2, 0, 3, 1, 4).unbind(dim=0)
        if attn.upcast_attention:
            q, k = q.float(), k.float()
        a = (q * attn.scale) @ k.transpose(-2, -1)
        if attn.enable_rpe:
            gc = point.grid_coord[order].reshape(-1, K, 3)
            a = a + attn.rpe(gc.unsqueeze(2) - gc.unsqueeze(1))
        a = a.float().softmax(dim=-1).to(qkv.dtype)
        feat = (a @ v).transpose(1, 2).reshape(-1, C)[inverse]
        return attn.proj(feat)

    def test_matches_original_math(self):
        K = 8
        for rpe in (False, True):
            point = _make_point([16, 24, 8, 32], channels=16, seed=4)
            point.serialization(order=["z", "hilbert"], depth=4)
            attn = SerializedAttention(
                16, 4, patch_size=K, order_index=1, enable_rpe=rpe, upcast_attention=rpe, upcast_softmax=rpe
            ).eval()
            ref = self._original_attention(attn, Point(point), K)
            out = attn(Point(point)).feat
            torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)

    def test_short_and_padded_patches(self):
        # events smaller than the patch attend over themselves; larger ones are split into patches
        K = 8
        point = _make_point([3, 13, 8, 21], channels=8, seed=5)
        point.serialization(order=["z"], depth=4)
        attn = SerializedAttention(8, 2, patch_size=K, upcast_attention=False, upcast_softmax=False).eval()
        out = attn(Point(point)).feat
        # event 0 (3 points) = plain full attention over its points
        H, C = 2, 8
        idx = torch.arange(3)
        qkv = attn.qkv(point.feat[idx])
        q, k, v = qkv.reshape(3, 3, H, C // H).permute(1, 2, 0, 3)
        ref = attn.proj(F.scaled_dot_product_attention(q, k, v).transpose(0, 1).reshape(3, C))
        torch.testing.assert_close(out[idx], ref, rtol=1e-5, atol=1e-5)
        self.assertTrue(torch.isfinite(out).all())


class TestPointTransformerV3(unittest.TestCase):
    def test_segmentation_mode_shapes_and_grad(self):
        model = PointTransformerV3(in_channels=8, **_SMALL)
        point = _make_point([20, 5, 37, 11], channels=8, seed=6)
        out = model(dict(point))
        self.assertEqual(out.feat.shape, (len(point.batch), _SMALL["dec_channels"][0]))
        out.feat.square().mean().backward()
        missing = [n for n, p in model.named_parameters() if p.grad is None]
        self.assertEqual(missing, [])

    def test_varlen_backend_rejects_rpe(self):
        with self.assertRaises(AssertionError):
            SerializedAttention(8, 2, 8, enable_rpe=True, upcast_attention=False, upcast_softmax=False,
                                attn_backend="varlen")


def _padded_inputs(counts, max_len, seed=0, pad_value=0.0):
    g = torch.Generator().manual_seed(seed)
    N = len(counts)
    points = torch.full((N, 3, max_len), pad_value)
    features = torch.full((N, 5, max_len), pad_value)
    mask = torch.zeros(N, 1, max_len)
    glob = torch.randn(N, 2, 1, generator=g)
    for i, n in enumerate(counts):
        # few distinct positions per event -> many points share a voxel
        pos = torch.randint(0, 6, (n, 3), generator=g).float() * 7.5 - 20.0
        points[i, :, :n] = pos.T
        features[i, :, :n] = torch.randn(5, n, generator=g)
        mask[i, :, :n] = 1
    return points, features, mask, glob


class TestPTv3EventRegressor(unittest.TestCase):
    def _model(self, **kwargs):
        torch.manual_seed(0)
        cfg = dict(input_dim=5, num_outputs=4, global_input_dim=2, grid_size=2.0, pooling="mean+max", **_SMALL)
        cfg.update(kwargs)
        return PTv3EventRegressor(**cfg)

    def test_padding_and_batch_invariance(self):
        model = self._model().eval()
        counts = [4, 30, 9, 17]
        points, features, mask, glob = _padded_inputs(counts, 32)
        with torch.no_grad():
            out = model(points, features, mask, glob)
            # different padding length and padding content
            p2, f2, m2, _ = _padded_inputs(counts, 45, pad_value=123.0)
            out2 = model(p2, f2, m2, glob)
            # each event alone
            single = torch.cat([model(points[i : i + 1], features[i : i + 1], mask[i : i + 1], glob[i : i + 1])
                                for i in range(len(counts))])
        self.assertEqual(out.shape, (4, 4))
        torch.testing.assert_close(out2, out, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(single, out, rtol=1e-4, atol=1e-5)

    def test_translation_invariance_of_voxelization(self):
        # the grid origin is per event: a global shift of the points by a multiple of grid_size
        # leaves the output unchanged when the absolute position is not part of the features
        model = self._model().eval()
        points, features, mask, glob = _padded_inputs([12, 25], 32)
        with torch.no_grad():
            torch.testing.assert_close(model(points + 20.0, features, mask, glob), model(points, features, mask, glob))

    def test_training_step(self):
        model = self._model(decoder=None) if False else self._model()
        model.train()
        points, features, mask, glob = _padded_inputs([5, 30, 9, 17, 22], 32, seed=3)
        out = model(points, features, mask, glob)
        out.square().mean().backward()
        grads = [p.grad for p in model.parameters()]
        self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in grads))

    def test_decoder_mode(self):
        model = self._model(cls_mode=False).eval()
        points, features, mask, glob = _padded_inputs([6, 19], 20)
        with torch.no_grad():
            self.assertEqual(model(points, features, mask, glob).shape, (2, 4))


if __name__ == "__main__":
    unittest.main()
