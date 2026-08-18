"""Unit tests for the L-GATr-slim port (weaver/nn/model/LGATrSlim.py).

Covers the tagger forward pass (shapes, padding invariance, permutation invariance,
``for_inference`` softmax) and the full ONNX export path via ``weaver.train.onnx``,
mirroring ``test_onnx_export.py``.
"""

import argparse
import os
import tempfile
import unittest

import numpy as np
import torch

try:
    import onnxruntime as ort

    _HAS_ORT = True
except ImportError:
    _HAS_ORT = False

from weaver import train as weaver_train
from weaver.utils.dataset import DataConfig
from weaver.utils.import_tools import import_module
from weaver.nn.model.LGATrSlim import (
    LGATrSlimTagger,
    _run_varlen_kernel,
    get_sparse_attention_kwargs,
)

try:
    import flash_attn  # noqa: F401

    _HAS_FLASH_ATTN = True
except ImportError:
    _HAS_FLASH_ATTN = False

try:
    import xformers.ops  # noqa: F401

    _HAS_XFORMERS = True
except ImportError:
    _HAS_XFORMERS = False

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA_CONFIG = os.path.join(_HERE, "data", "JetClass_full.yaml")
_NETWORK_CONFIG = os.path.join(_HERE, "networks", "example_LGATrSlim.py")

# small config for fast tests
_SMALL_NET = dict(hidden_v_channels=8, hidden_s_channels=16, num_blocks=2, num_heads=2)


def _make_inputs(batch=3, num_features=17, seq_len=20, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(batch, num_features, seq_len, generator=g)
    px, py, pz = (torch.randn(batch, 1, seq_len, generator=g) + 0.5 for _ in range(3))
    m = torch.rand(batch, 1, seq_len, generator=g) * 0.3 + 0.1
    v = torch.cat([px, py, pz, torch.sqrt(px**2 + py**2 + pz**2 + m**2)], dim=1)
    mask = torch.zeros(batch, 1, seq_len)
    for i in range(batch):
        mask[i, 0, : max(1, seq_len - 6 * i)] = 1
    return x, v, mask


class LGATrSlimTaggerTest(unittest.TestCase):
    def _make_tagger(self, **kwargs):
        cfg = dict(input_dim=17, num_classes=10, trim=False, **_SMALL_NET)
        cfg.update(kwargs)
        torch.manual_seed(0)
        model = LGATrSlimTagger(**cfg)
        model.eval()
        return model

    def test_forward_shape(self):
        model = self._make_tagger()
        x, v, mask = _make_inputs()
        with torch.no_grad():
            out = model(x, v, mask)
        self.assertEqual(out.shape, (3, 10))
        self.assertFalse(torch.isnan(out).any())

    def test_padding_invariance(self):
        model = self._make_tagger()
        x, v, mask = _make_inputs()
        m = mask.squeeze(1).bool().unsqueeze(1)
        x2 = torch.where(m, x, torch.full_like(x, 123.0))
        v2 = torch.where(m, v, torch.full_like(v, -77.0))
        with torch.no_grad():
            out = model(x, v, mask)
            out2 = model(x2, v2, mask)
        torch.testing.assert_close(out, out2, rtol=0, atol=1e-5)

    def test_permutation_invariance(self):
        model = self._make_tagger()
        x, v, mask = _make_inputs()
        perm = torch.randperm(x.size(-1))
        # permute only jet 0, which is fully valid
        x2, v2 = x.clone(), v.clone()
        x2[0] = x[0][:, perm]
        v2[0] = v[0][:, perm]
        with torch.no_grad():
            out = model(x, v, mask)
            out2 = model(x2, v2, mask)
        torch.testing.assert_close(out[0], out2[0], rtol=1e-4, atol=1e-4)

    def test_for_inference_softmax(self):
        model = self._make_tagger()
        model_inf = self._make_tagger(for_inference=True)
        model_inf.load_state_dict(model.state_dict())
        x, v, mask = _make_inputs()
        with torch.no_grad():
            logits = model(x, v, mask)
            probs = model_inf(x, v, mask)
        torch.testing.assert_close(probs, torch.softmax(logits, dim=1), rtol=1e-5, atol=1e-6)

    def test_mean_aggregation(self):
        model = self._make_tagger(mean_aggregation=True)
        x, v, mask = _make_inputs()
        with torch.no_grad():
            out = model(x, v, mask)
        self.assertEqual(out.shape, (3, 10))
        self.assertFalse(torch.isnan(out).any())

    def test_auxiliary_scalars(self):
        # by default no kinematic features are computed; they can be switched on
        x, v, mask = _make_inputs()
        for aux, num_aux in [(None, 0), ("all", 7), ("zinvariant", 5), ("so3invariant", 2)]:
            with self.subTest(auxiliary_scalars=aux):
                model = self._make_tagger(auxiliary_scalars=aux)
                self.assertEqual(
                    model.net.linear_in.linear_s.in_features, num_aux + 17 + 1
                )
                with torch.no_grad():
                    out = model(x, v, mask)
                self.assertEqual(out.shape, (3, 10))
                self.assertFalse(torch.isnan(out).any())

    def test_auxiliary_scalars_padding_invariance(self):
        model = self._make_tagger(auxiliary_scalars="all")
        x, v, mask = _make_inputs()
        m = mask.squeeze(1).bool().unsqueeze(1)
        x2 = torch.where(m, x, torch.full_like(x, 123.0))
        v2 = torch.where(m, v, torch.full_like(v, -77.0))
        with torch.no_grad():
            out = model(x, v, mask)
            out2 = model(x2, v2, mask)
        torch.testing.assert_close(out, out2, rtol=0, atol=1e-5)

    def test_backward(self):
        model = self._make_tagger()
        model.train()
        x, v, mask = _make_inputs()
        out = model(x, v, mask)
        loss = torch.nn.functional.cross_entropy(out, torch.randint(0, 10, (x.size(0),)))
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.requires_grad]
        self.assertTrue(all(g is not None for g in grads))
        self.assertTrue(all(torch.isfinite(g).all() for g in grads if g is not None))


class LGATrSlimPackedAttentionTest(unittest.TestCase):
    """Packed (sparse) attention path tests.

    On CPU the varlen backends fall back to a materialized block-diagonal SDPA mask, so
    these tests exercise the full packed layout (padding removal, per-event global
    tokens, block-diagonal attention, segment-wise readout) without a GPU.
    """

    def _make_pair(self, backend, **kwargs):
        cfg = dict(input_dim=17, num_classes=10, trim=False, **_SMALL_NET)
        cfg.update(kwargs)
        torch.manual_seed(0)
        dense = LGATrSlimTagger(**cfg)
        dense.eval()
        packed = LGATrSlimTagger(**cfg, attention_backend=backend)
        packed.load_state_dict(dense.state_dict())
        packed.eval()
        return dense, packed

    def test_invalid_backend(self):
        with self.assertRaises(ValueError):
            LGATrSlimTagger(input_dim=17, num_classes=10, attention_backend="flex")

    def test_packed_matches_dense(self):
        for mean_aggregation in [False, True]:
            with self.subTest(mean_aggregation=mean_aggregation):
                dense, packed = self._make_pair("varlen", mean_aggregation=mean_aggregation)
                x, v, mask = _make_inputs()
                with torch.no_grad():
                    out_dense = dense(x, v, mask)
                    out_packed = packed(x, v, mask)
                torch.testing.assert_close(out_dense, out_packed, rtol=1e-4, atol=1e-5)

    def test_packed_backward(self):
        _, packed = self._make_pair("varlen")
        packed.train()
        x, v, mask = _make_inputs()
        out = packed(x, v, mask)
        loss = torch.nn.functional.cross_entropy(out, torch.randint(0, 10, (x.size(0),)))
        loss.backward()
        grads = [p.grad for p in packed.parameters() if p.requires_grad]
        self.assertTrue(all(g is not None for g in grads))
        self.assertTrue(all(torch.isfinite(g).all() for g in grads if g is not None))

    def test_varlen_kernel_wrapper(self):
        """Check the kernel wrapper (packing reshape, head-dim padding, dtype casting)
        against a reference block-diagonal SDPA, using a stub varlen kernel."""

        channels = 6  # head dim deliberately not a multiple of 8

        def stub_kernel(q, k, v, cu_seq_q=None, cu_seq_k=None, max_q=None, max_k=None, scale=None):
            # emulate a varlen kernel on CPU: per-segment SDPA on (tokens, heads, C)
            self.assertEqual(cu_seq_q.dtype, torch.int32)
            self.assertLessEqual(int((cu_seq_q[1:] - cu_seq_q[:-1]).max()), max_q)
            self.assertEqual(q.shape[-1] % 8, 0)  # head dim padded to a multiple of 8
            self.assertIn(q.dtype, (torch.float16, torch.bfloat16))
            # the softmax scale must correspond to the un-padded head dim
            self.assertAlmostEqual(scale, channels**-0.5)
            out = torch.empty_like(q)
            for a, b in zip(cu_seq_q[:-1].tolist(), cu_seq_q[1:].tolist()):
                seg = torch.nn.functional.scaled_dot_product_attention(
                    q[a:b].transpose(0, 1),
                    k[a:b].transpose(0, 1),
                    v[a:b].transpose(0, 1),
                    scale=scale,
                )
                out[a:b] = seg.transpose(0, 1)
            return out

        torch.manual_seed(0)
        heads = 2
        seqlens = [4, 7, 1]
        total = sum(seqlens)
        cu = torch.tensor([0] + list(np.cumsum(seqlens)), dtype=torch.int32)
        q, k, v = (torch.randn(1, heads, total, channels) for _ in range(3))

        out = _run_varlen_kernel(
            stub_kernel, q, k, v,
            dict(cu_seq_q=cu, cu_seq_k=cu, max_q=max(seqlens), max_k=max(seqlens)),
        )
        self.assertEqual(out.shape, q.shape)
        self.assertEqual(out.dtype, q.dtype)

        batch = torch.arange(len(seqlens)).repeat_interleave(torch.tensor(seqlens))
        attn_mask = batch.unsqueeze(0) == batch.unsqueeze(1)
        ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        # the stub runs in bf16/fp16 inside the wrapper, hence the loose tolerance
        torch.testing.assert_close(out, ref, rtol=0.05, atol=0.05)

    def test_sparse_attention_kwargs(self):
        # CPU: materialized block-diagonal mask fallback
        ptr = torch.tensor([0, 3, 5])
        batch = torch.tensor([0, 0, 0, 1, 1])
        kwargs = get_sparse_attention_kwargs(ptr, batch, maxlen=3, attention_backend="varlen")
        expected = batch.unsqueeze(0) == batch.unsqueeze(1)
        self.assertEqual(list(kwargs), ["attn_mask"])
        self.assertTrue(torch.equal(kwargs["attn_mask"], expected))

        # non-CPU: cu_seqlens kwargs of the requested kernel (meta device stands in for CUDA)
        ptr_meta = ptr.to("meta")
        for backend, keys in [
            ("varlen", ["cu_seq_q", "cu_seq_k", "max_q", "max_k"]),
            ("flash", ["cu_seqlens_q", "cu_seqlens_k", "max_seqlen_q", "max_seqlen_k"]),
        ]:
            with self.subTest(backend=backend):
                kwargs = get_sparse_attention_kwargs(
                    ptr_meta, batch.to("meta"), maxlen=3, attention_backend=backend
                )
                self.assertEqual(sorted(kwargs), sorted(keys))
                self.assertEqual(kwargs[keys[0]].dtype, torch.int32)
                self.assertEqual(kwargs[keys[2]], 3)
        with self.assertRaises(ValueError):
            get_sparse_attention_kwargs(
                ptr_meta, batch.to("meta"), maxlen=3, attention_backend="native"
            )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for the varlen kernel")
    def test_varlen_backend_cuda(self):
        dense, packed = self._make_pair("varlen")
        dense, packed = dense.cuda(), packed.cuda()
        x, v, mask = (t.cuda() for t in _make_inputs())
        with torch.no_grad():
            out_dense = dense(x, v, mask)
            out_packed = packed(x, v, mask)
        # the varlen kernel runs in half precision
        torch.testing.assert_close(out_dense, out_packed, rtol=2e-2, atol=2e-2)

    @unittest.skipUnless(
        torch.cuda.is_available() and _HAS_FLASH_ATTN,
        "CUDA and flash-attn are required for the flash backend",
    )
    def test_flash_backend_cuda(self):
        dense, packed = self._make_pair("flash")
        dense, packed = dense.cuda(), packed.cuda()
        x, v, mask = (t.cuda() for t in _make_inputs())
        with torch.no_grad():
            out_dense = dense(x, v, mask)
            out_packed = packed(x, v, mask)
        torch.testing.assert_close(out_dense, out_packed, rtol=2e-2, atol=2e-2)

    @unittest.skipUnless(
        torch.cuda.is_available() and _HAS_XFORMERS,
        "CUDA and xformers are required for the xformers backend",
    )
    def test_xformers_backend_cuda(self):
        dense, packed = self._make_pair("xformers")
        dense, packed = dense.cuda(), packed.cuda()
        x, v, mask = (t.cuda() for t in _make_inputs())
        with torch.no_grad():
            out_dense = dense(x, v, mask)
            out_packed = packed(x, v, mask)
        # the xformers kernel runs in half precision
        torch.testing.assert_close(out_dense, out_packed, rtol=2e-2, atol=2e-2)


@unittest.skipUnless(_HAS_ORT, "onnxruntime is required for ONNX export tests")
class LGATrSlimOnnxExportTest(unittest.TestCase):
    def test_export(self):
        data_config = DataConfig.load(_DATA_CONFIG, load_observers=False, load_reweight_info=False)
        net = import_module(_NETWORK_CONFIG, name="_lgatr_onnx_test_net")

        with tempfile.TemporaryDirectory() as workdir:
            args = argparse.Namespace(
                data_config=_DATA_CONFIG,
                network_config=_NETWORK_CONFIG,
                network_option=[[k, repr(val)] for k, val in _SMALL_NET.items()],
                model_prefix=os.path.join(workdir, "net.pt"),
                export_onnx=os.path.join(workdir, "model.onnx"),
                onnx_opset=15,
                use_amp=False,
                compile=False,
                load_model_weights=None,
                exclude_model_weights=None,
                freeze_model_weights=None,
                regression_mode=False,
            )

            train_model, _ = net.get_model(data_config, **_SMALL_NET)
            train_model.eval()
            torch.save(train_model.state_dict(), args.model_prefix)

            weaver_train.onnx(args)
            self.assertTrue(os.path.isfile(args.export_onnx))

            sess = ort.InferenceSession(args.export_onnx, providers=["CPUExecutionProvider"])
            seq_len = data_config.input_shapes["pf_mask"][2]
            num_features = data_config.input_shapes["pf_features"][1]
            # exercise the declared-dynamic axes: vary batch size and sequence length
            for batch, plen in [(1, seq_len), (4, seq_len), (3, seq_len // 2)]:
                with self.subTest(batch=batch, seq_len=plen):
                    x, v, mask = _make_inputs(batch=batch, num_features=num_features, seq_len=plen, seed=1)
                    inp = {"pf_features": x, "pf_vectors": v, "pf_mask": mask}
                    inp["pf_points"] = torch.randn(batch, data_config.input_shapes["pf_points"][1], plen)
                    with torch.no_grad():
                        logits = train_model(*[inp[k] for k in data_config.input_names])
                        ref = torch.softmax(logits, dim=1).numpy()
                    feed = {i.name: inp[i.name].numpy().astype(np.float32) for i in sess.get_inputs()}
                    out = sess.run(None, feed)[0]
                    self.assertFalse(np.isnan(out).any(), msg="ONNX output contains NaNs")
                    np.testing.assert_allclose(
                        out, ref, rtol=1e-3, atol=1e-4, err_msg="ONNX output differs from PyTorch"
                    )


if __name__ == "__main__":
    unittest.main()
