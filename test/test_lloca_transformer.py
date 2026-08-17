"""Unit tests for the LLoCa-Transformer port (weaver/nn/model/LLoCaTransformer.py).

Covers the tagger forward pass (shapes, padding invariance, permutation invariance,
``for_inference`` softmax, Lorentz invariance without spurions), cross-checks of the
dense reimplementation against the upstream ``lloca`` package (if installed), and the
full ONNX export path via ``weaver.train.onnx``, mirroring ``test_lgatr_slim.py``.
"""

import argparse
import math
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

from weaver import train as weaver_train
from weaver.utils.dataset import DataConfig
from weaver.utils.import_tools import import_module
from weaver.nn.model.LLoCaTransformer import (
    LLoCaTransformer,
    LLoCaTransformerTagger,
    LearnedPDFrames,
    MLPVectors,
)

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA_CONFIG = os.path.join(_HERE, "data", "JetClass_full.yaml")
_NETWORK_CONFIG = os.path.join(_HERE, "networks", "example_LLoCaTransformer.py")

# small config for fast tests
_SMALL_NET = dict(
    attn_reps="4x0n+2x1n", num_heads=2, num_blocks=2, equivectors_hidden_channels=16
)
# onnxruntime has no float64 kernels for some ops (e.g. Atan, the optimizer's FusedMatMul),
# so ONNX export requires the float32 paths
_ONNX_NET = dict(_SMALL_NET, momentum_float64=False, ortho_use_float64=False)

_ORTHO_KWARGS = dict(
    use_float64=True,
    method="gramschmidt",
    eps_norm=1e-15,
    eps_reg=1e-16,
    eps_reg_lightlike=1e-16,
    checks=False,
)


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


def _apply_lorentz(v, trafo):
    """Apply an (E, px, py, pz)-convention Lorentz matrix to v of shape (N, 4, P)."""
    p = v.transpose(1, 2)[..., [3, 0, 1, 2]]  # (N, P, 4) energy-first
    p = torch.einsum("ij,npj->npi", trafo.to(p.dtype), p)
    return p[..., [1, 2, 3, 0]].transpose(1, 2)


def _rotation_z(angle):
    c, s = math.cos(angle), math.sin(angle)
    return torch.tensor(
        [[1, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, 1]], dtype=torch.float64
    )


def _rotation_x(angle):
    c, s = math.cos(angle), math.sin(angle)
    return torch.tensor(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, c, -s], [0, 0, s, c]], dtype=torch.float64
    )


def _boost_z(beta):
    gamma = 1.0 / math.sqrt(1.0 - beta**2)
    return torch.tensor(
        [
            [gamma, 0, 0, gamma * beta],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [gamma * beta, 0, 0, gamma],
        ],
        dtype=torch.float64,
    )


class LLoCaTransformerTaggerTest(unittest.TestCase):
    def _make_tagger(self, **kwargs):
        cfg = dict(input_dim=17, num_classes=10, trim=False, **_SMALL_NET)
        cfg.update(kwargs)
        torch.manual_seed(0)
        model = LLoCaTransformerTagger(**cfg)
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

    def test_lorentz_invariance(self):
        # without spurions and without global-frame kinematic inputs to the frames-net,
        # the model is exactly Lorentz-invariant; mean aggregation is required because the
        # global class token carries a fixed identity frame (as in the upstream wrapper),
        # which deliberately breaks exact invariance of token <-> particle attention
        model = self._make_tagger(
            beam_reference=None,
            add_time_reference=False,
            auxiliary_scalars=None,
            mean_aggregation=True,
        )
        x, v, mask = _make_inputs()
        with torch.no_grad():
            out = model(x, v, mask)
            for trafo in [
                _rotation_z(0.7),
                _rotation_x(-1.2) @ _rotation_z(0.4),
                _boost_z(0.4),
                _rotation_x(0.5) @ _boost_z(-0.3) @ _rotation_z(2.0),
            ]:
                out2 = model(x, _apply_lorentz(v, trafo), mask)
                torch.testing.assert_close(out, out2, rtol=1e-3, atol=1e-3)

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

    def test_backward_gamma_clamp(self):
        model = self._make_tagger(gamma_max=5.0, gamma_hardness=10.0)
        model.train()
        x, v, mask = _make_inputs()
        out = model(x, v, mask)
        self.assertTrue(torch.isfinite(out).all())
        out.sum().backward()
        grads = [p.grad for p in model.parameters() if p.requires_grad]
        self.assertTrue(all(torch.isfinite(g).all() for g in grads if g is not None))


class LLoCaPackedAttentionTest(unittest.TestCase):
    """Packed (sparse) attention path tests.

    On CPU the varlen backends fall back to a materialized block-diagonal SDPA mask, so
    these tests exercise the full packed layout (padding removal, per-event global
    tokens with identity frames, token-resolved reference momenta, block-diagonal
    attention, segment-wise readout) without a GPU.
    """

    def _make_pair(self, backend, **kwargs):
        cfg = dict(input_dim=17, num_classes=10, trim=False, **_SMALL_NET)
        cfg.update(kwargs)
        torch.manual_seed(0)
        dense = LLoCaTransformerTagger(**cfg)
        dense.eval()
        packed = LLoCaTransformerTagger(**cfg, attention_backend=backend)
        packed.load_state_dict(dense.state_dict())
        packed.eval()
        return dense, packed

    def test_invalid_backend(self):
        with self.assertRaises(ValueError):
            LLoCaTransformerTagger(input_dim=17, num_classes=10, attention_backend="xformers")

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


@unittest.skipUnless(_HAS_LLOCA, "the upstream lloca package is required for cross-checks")
class UpstreamCrossCheckTest(unittest.TestCase):
    """Compare the dense masked reimplementation against the upstream sparse lloca code."""

    def _dense_and_sparse_inputs(self, num_scalars=4, seed=0):
        g = torch.Generator().manual_seed(seed)
        batch, seq_len = 3, 10
        px, py, pz = (torch.randn(batch, seq_len, generator=g) + 0.5 for _ in range(3))
        m = torch.rand(batch, seq_len, generator=g) * 0.3 + 0.1
        e = torch.sqrt(px**2 + py**2 + pz**2 + m**2)
        fourmomenta = torch.stack([e, px, py, pz], dim=-1).to(torch.float64)
        scalars = torch.randn(batch, seq_len, num_scalars, generator=g)
        lengths = [seq_len, seq_len - 3, seq_len - 6]
        mask = torch.zeros(batch, seq_len, dtype=torch.bool)
        for i, n in enumerate(lengths):
            mask[i, :n] = True
        fourmomenta = fourmomenta * mask.unsqueeze(-1)
        scalars = scalars * mask.unsqueeze(-1)

        fm_sparse = fourmomenta[mask]
        s_sparse = scalars[mask]
        ptr = torch.tensor([0] + list(np.cumsum(lengths)))
        return fourmomenta, scalars, mask, fm_sparse, s_sparse, ptr

    def test_equivectors_match(self):
        from lloca.equivectors.mlp import MLPVectors as UpstreamMLPVectors

        num_scalars = 4
        torch.manual_seed(0)
        dense = MLPVectors(
            n_vectors=3, num_scalars=num_scalars, hidden_channels=16, num_layers_mlp=2
        )
        upstream = UpstreamMLPVectors(
            n_vectors=3, num_scalars=num_scalars, hidden_channels=16, num_layers_mlp=2
        )

        (fourmomenta, scalars, mask, fm_sparse, s_sparse, ptr) = self._dense_and_sparse_inputs(
            num_scalars
        )
        dense.init_standardization(fourmomenta, mask)
        upstream.load_state_dict(dense.state_dict())

        with torch.no_grad():
            vecs_dense = dense(fourmomenta, scalars, mask)
            vecs_upstream = upstream(fm_sparse, scalars=s_sparse, ptr=ptr)
        torch.testing.assert_close(
            vecs_dense[mask], vecs_upstream.to(vecs_dense.dtype), rtol=1e-5, atol=1e-6
        )

    def test_frames_match(self):
        from functools import partial

        from lloca.equivectors.mlp import MLPVectors as UpstreamMLPVectors
        from lloca.framesnet.equi_frames import LearnedPDFrames as UpstreamLearnedPDFrames

        num_scalars = 4
        torch.manual_seed(0)
        dense = LearnedPDFrames(
            partial(MLPVectors, num_scalars=num_scalars, hidden_channels=16, num_layers_mlp=2),
            mass_reg=5e-3,
            ortho_kwargs=_ORTHO_KWARGS,
        )
        upstream = UpstreamLearnedPDFrames(
            partial(
                UpstreamMLPVectors,
                num_scalars=num_scalars,
                hidden_channels=16,
                num_layers_mlp=2,
            ),
            mass_reg=5e-3,
            ortho_kwargs=_ORTHO_KWARGS,
        )

        (fourmomenta, scalars, mask, fm_sparse, s_sparse, ptr) = self._dense_and_sparse_inputs(
            num_scalars
        )
        dense.init_standardization(fourmomenta, mask)
        upstream.load_state_dict(dense.state_dict())

        with torch.no_grad():
            frames_dense = dense(fourmomenta, scalars=scalars, mask=mask)
            frames_upstream = upstream(fm_sparse, scalars=s_sparse, ptr=ptr)
        torch.testing.assert_close(
            frames_dense.matrices[mask],
            frames_upstream.matrices.to(frames_dense.dtype),
            rtol=1e-4,
            atol=1e-5,
        )

    def test_transformer_match(self):
        from functools import partial

        from lloca.backbone.transformer_v2 import Transformer as UpstreamTransformer
        from lloca.framesnet.frames import Frames as UpstreamFrames

        in_channels, out_channels = 6, 3
        torch.manual_seed(0)
        # the upstream main branch has no preserve_variance option and parameter-free norms
        mine = LLoCaTransformer(
            in_channels=in_channels,
            attn_reps="4x0n+2x1n",
            out_channels=out_channels,
            num_blocks=2,
            num_heads=2,
            preserve_variance=False,
            elementwise_affine=False,
        )
        upstream = UpstreamTransformer(
            in_channels=in_channels,
            attn_reps="4x0n+2x1n",
            out_channels=out_channels,
            num_blocks=2,
            num_heads=2,
        )
        upstream.load_state_dict(mine.state_dict())

        # frames from the dense frames-net on a fully-valid batch
        num_scalars = 4
        framesnet = LearnedPDFrames(
            partial(MLPVectors, num_scalars=num_scalars, hidden_channels=16, num_layers_mlp=2),
            mass_reg=5e-3,
            ortho_kwargs=_ORTHO_KWARGS,
        )
        g = torch.Generator().manual_seed(1)
        batch, seq_len = 2, 8
        px, py, pz = (torch.randn(batch, seq_len, generator=g) + 0.5 for _ in range(3))
        m = torch.rand(batch, seq_len, generator=g) * 0.3 + 0.1
        e = torch.sqrt(px**2 + py**2 + pz**2 + m**2)
        fourmomenta = torch.stack([e, px, py, pz], dim=-1).to(torch.float64)
        scalars = torch.randn(batch, seq_len, num_scalars, generator=g)
        mask = torch.ones(batch, seq_len, dtype=torch.bool)
        framesnet.init_standardization(fourmomenta, mask)
        with torch.no_grad():
            frames_mine = framesnet(fourmomenta, scalars=scalars, mask=mask)
        frames_mine.to(dtype=torch.float32)
        frames_upstream = UpstreamFrames(matrices=frames_mine.matrices.clone())

        inputs = torch.randn(batch, seq_len, in_channels, generator=g)
        with torch.no_grad():
            out_mine = mine(inputs, frames=frames_mine)
            out_upstream = upstream(inputs, frames_upstream)
        torch.testing.assert_close(out_mine, out_upstream, rtol=1e-4, atol=1e-5)


@unittest.skipUnless(_HAS_ORT, "onnxruntime is required for ONNX export tests")
class LLoCaTransformerOnnxExportTest(unittest.TestCase):
    def test_export(self):
        data_config = DataConfig.load(_DATA_CONFIG, load_observers=False, load_reweight_info=False)
        net = import_module(_NETWORK_CONFIG, name="_lloca_onnx_test_net")

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
            # initialize the frames-net edge standardization before export
            x, v, mask = _make_inputs(
                num_features=len(data_config.input_dicts["pf_features"]), seed=2
            )
            with torch.no_grad():
                train_model(None, x, v, mask)
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
