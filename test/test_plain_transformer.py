"""Unit tests for the plain-transformer port (weaver/nn/model/PlainTransformer.py).

Covers the tagger forward pass (shapes, padding invariance, permutation invariance,
``for_inference`` softmax, aggregation modes), the packed attention path, a cross-check
of the identity-frames backbone against the upstream ``lloca`` package (if installed),
and the full ONNX export path via ``weaver.train.onnx``, mirroring
``test_lloca_transformer.py``.
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

try:
    import lloca.backbone.transformer_v2  # noqa: F401

    _HAS_LLOCA = True
except ImportError:
    _HAS_LLOCA = False

try:
    import xformers.ops  # noqa: F401

    _HAS_XFORMERS = True
except ImportError:
    _HAS_XFORMERS = False

from weaver import train as weaver_train
from weaver.utils.dataset import DataConfig
from weaver.utils.import_tools import import_module
from weaver.nn.model.LLoCaTransformer import Frames, LLoCaTransformer
from weaver.nn.model.PlainTransformer import PlainTransformerTagger

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA_CONFIG = os.path.join(_HERE, "data", "JetClass_full.yaml")
_NETWORK_CONFIG = os.path.join(_HERE, "networks", "example_PlainTransformer.py")

# small config for fast tests
_SMALL_NET = dict(embed_dim=32, num_heads=2, num_blocks=2)
# onnxruntime has no float64 kernels for some ops (e.g. Atan), so ONNX export requires
# the float32 path
_ONNX_NET = dict(_SMALL_NET, momentum_float64=False)


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


class PlainTransformerTaggerTest(unittest.TestCase):
    def _make_tagger(self, **kwargs):
        cfg = dict(input_dim=17, num_classes=10, trim=False, **_SMALL_NET)
        cfg.update(kwargs)
        torch.manual_seed(0)
        model = PlainTransformerTagger(**cfg)
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

    def test_no_register_tokens(self):
        model = self._make_tagger(num_register_tokens=0)
        x, v, mask = _make_inputs()
        with torch.no_grad():
            out = model(x, v, mask)
        self.assertEqual(out.shape, (3, 10))
        self.assertFalse(torch.isnan(out).any())

    def test_momentum_float32(self):
        model = self._make_tagger(momentum_float64=False)
        x, v, mask = _make_inputs()
        with torch.no_grad():
            out = model(x, v, mask)
        self.assertEqual(out.shape, (3, 10))
        self.assertFalse(torch.isnan(out).any())

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


class PlainPackedAttentionTest(unittest.TestCase):
    """Packed (sparse) attention path tests.

    On CPU the varlen backends fall back to a materialized block-diagonal SDPA mask, so
    these tests exercise the full packed layout (padding removal, per-event global
    tokens, block-diagonal attention, segment-wise readout) without a GPU.
    """

    def _make_pair(self, backend, **kwargs):
        cfg = dict(input_dim=17, num_classes=10, trim=False, **_SMALL_NET)
        cfg.update(kwargs)
        torch.manual_seed(0)
        dense = PlainTransformerTagger(**cfg)
        dense.eval()
        packed = PlainTransformerTagger(**cfg, attention_backend=backend)
        packed.load_state_dict(dense.state_dict())
        packed.eval()
        return dense, packed

    def test_invalid_backend(self):
        with self.assertRaises(ValueError):
            PlainTransformerTagger(input_dim=17, num_classes=10, attention_backend="flex")

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


@unittest.skipUnless(_HAS_LLOCA, "the upstream lloca package is required for cross-checks")
class UpstreamIdentityCrossCheckTest(unittest.TestCase):
    """Compare the identity-frames backbone against the upstream lloca transformer."""

    def test_transformer_match(self):
        from lloca.backbone.transformer_v2 import Transformer as UpstreamTransformer
        from lloca.framesnet.frames import Frames as UpstreamFrames

        in_channels, out_channels = 6, 3
        torch.manual_seed(0)
        # the upstream main branch has no preserve_variance option and parameter-free norms
        mine = LLoCaTransformer(
            in_channels=in_channels,
            attn_reps="16x0n",
            out_channels=out_channels,
            num_blocks=2,
            num_heads=2,
            preserve_variance=False,
            elementwise_affine=False,
        )
        upstream = UpstreamTransformer(
            in_channels=in_channels,
            attn_reps="16x0n",
            out_channels=out_channels,
            num_blocks=2,
            num_heads=2,
        )
        upstream.load_state_dict(mine.state_dict())

        g = torch.Generator().manual_seed(1)
        batch, seq_len = 2, 8
        inputs = torch.randn(batch, seq_len, in_channels, generator=g)
        frames_mine = Frames(
            is_identity=True, device=inputs.device, dtype=inputs.dtype,
            shape=inputs.shape[:-1],
        )
        frames_upstream = UpstreamFrames(
            is_identity=True, device=inputs.device, dtype=inputs.dtype,
            shape=inputs.shape[:-1],
        )
        with torch.no_grad():
            out_mine = mine(inputs, frames=frames_mine)
            out_upstream = upstream(inputs, frames_upstream)
        torch.testing.assert_close(out_mine, out_upstream, rtol=1e-5, atol=1e-6)


@unittest.skipUnless(_HAS_ORT, "onnxruntime is required for ONNX export tests")
class PlainTransformerOnnxExportTest(unittest.TestCase):
    def test_export(self):
        data_config = DataConfig.load(_DATA_CONFIG, load_observers=False, load_reweight_info=False)
        net = import_module(_NETWORK_CONFIG, name="_plain_onnx_test_net")

        with tempfile.TemporaryDirectory() as workdir:
            args = argparse.Namespace(
                data_config=_DATA_CONFIG,
                network_config=_NETWORK_CONFIG,
                # values go through ast.literal_eval, so strings must stay quoted
                network_option=[[k, repr(val)] for k, val in _ONNX_NET.items()],
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

            train_model, _ = net.get_model(data_config, **_ONNX_NET)
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
                    x, v, mask = _make_inputs(
                        batch=batch, num_features=num_features, seq_len=plen, seed=1
                    )
                    inp = {"pf_features": x, "pf_vectors": v, "pf_mask": mask}
                    inp["pf_points"] = torch.randn(
                        batch, data_config.input_shapes["pf_points"][1], plen
                    )
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
