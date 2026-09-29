"""Unit tests for the L-GATr-slim port (weaver/nn/model/LGATrSlim.py).

Covers the tagger forward pass (shapes, padding invariance, permutation invariance,
``for_inference`` softmax), the exactness of the light-cone frame, and the full ONNX export
path via ``weaver.train.onnx``, mirroring ``test_onnx_export.py``.
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
    _flex_block_size,
    _half_cast_dtype,
    _run_varlen_kernel,
    _weighted_directions,
    collinearity_weights,
    get_collinear_axis,
    get_lightcone_frame,
    get_principal_axis,
    get_sparse_attention_kwargs,
    leading_eigenvector,
    lightcone_frame_from_axis,
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


class LGATrSlimLightconeTest(unittest.TestCase):
    """``vector_coord="lightcone"`` is an exact change of basis.

    The per-jet map to light-cone coordinates is orthogonal and turns the Minkowski metric
    into the light-cone metric, so with identical weights the light-cone model must
    reproduce the Cartesian one -- in fp64 to rounding, outputs and gradients alike. Only
    the conditioning (and hence half-precision accuracy) differs.
    """

    def _make_pair(self, **kwargs):
        cfg = dict(input_dim=17, num_classes=10, trim=False, **_SMALL_NET)
        cfg.update(kwargs)
        torch.manual_seed(0)
        cartesian = LGATrSlimTagger(**cfg, vector_coord="cartesian").double()
        with torch.no_grad():
            # the "small" init of the q/k/v projection hides most of the attention
            for block in cartesian.net.blocks:
                block.attention.linear_in.weight_v.mul_(10)
                block.attention.linear_in.linear_s.weight.mul_(10)
        lightcone = LGATrSlimTagger(**cfg, vector_coord="lightcone").double()
        lightcone.load_state_dict(cartesian.state_dict())
        return cartesian, lightcone

    def test_frame_is_orthogonal_with_lightcone_metric(self):
        _, v, mask = _make_inputs()
        fourmomenta = v.double().transpose(1, 2)[..., [3, 0, 1, 2]]  # (E, px, py, pz)
        frame = get_lightcone_frame(fourmomenta, mask.squeeze(1))
        eye = torch.eye(4, dtype=torch.float64).expand_as(frame)
        torch.testing.assert_close(frame @ frame.transpose(-1, -2), eye)
        eta = torch.diag(torch.tensor([1.0, -1.0, -1.0, -1.0], dtype=torch.float64))
        eta_lightcone = torch.tensor(
            [[0.0, 1, 0, 0], [1, 0, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]], dtype=torch.float64
        )
        torch.testing.assert_close(
            frame @ eta @ frame.transpose(-1, -2), eta_lightcone.expand_as(frame)
        )

    def test_matches_cartesian_fp64(self):
        x, v, mask = (t.double() for t in _make_inputs())
        for backend in ["native", "varlen"]:
            for mean_aggregation in [False, True]:
                with self.subTest(backend=backend, mean_aggregation=mean_aggregation):
                    cartesian, lightcone = self._make_pair(
                        attention_backend=backend, mean_aggregation=mean_aggregation
                    )
                    results = []
                    for model in (cartesian, lightcone):
                        out = model(x, v, mask)
                        out.square().sum().backward()
                        grads = {n: p.grad for n, p in model.named_parameters() if p.grad is not None}
                        results.append((out.detach(), grads))
                    (out_c, grads_c), (out_l, grads_l) = results
                    torch.testing.assert_close(out_l, out_c, rtol=1e-9, atol=1e-10)
                    self.assertEqual(grads_l.keys(), grads_c.keys())
                    for name in grads_c:
                        torch.testing.assert_close(
                            grads_l[name], grads_c[name], rtol=1e-8, atol=1e-10, msg=name
                        )


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
            LGATrSlimTagger(input_dim=17, num_classes=10, attention_backend="nonesuch")

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

    def test_half_cast_dtype(self):
        """Outside autocast the packed kernels fall back to fp16, not bf16.

        Both underflow-free bf16 and mantissa-rich fp16 were measured on a 6-epoch
        JetClass run: fp16 won by ~1 pp of val accuracy in the packed layout, because
        bf16's coarser mantissa hurts the forward values more than fp16's backward
        underflow hurts (the entries that underflow carry ~1% of the gradient norm).
        """
        cpu = torch.device("cpu")
        self.assertEqual(_half_cast_dtype(cpu), torch.float16)

        # inside autocast the surrounding autocast dtype wins
        with torch.autocast("cpu", dtype=torch.bfloat16):
            self.assertEqual(_half_cast_dtype(cpu), torch.bfloat16)

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

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for flex_attention")
    def test_flex_backend_cuda(self):
        """flex_attention has an fp32 path, so unlike the flash-derived packed backends
        it must reproduce the dense fp32 result to tight tolerance, not to 2e-2."""
        dense, packed = self._make_pair("flex")
        dense, packed = dense.cuda(), packed.cuda()
        x, v, mask = (t.cuda() for t in _make_inputs())
        with torch.no_grad():
            out_dense = dense(x, v, mask)
            out_packed = packed(x, v, mask)
        torch.testing.assert_close(out_dense, out_packed, rtol=1e-4, atol=1e-5)

    def test_flex_head_dim_matches_attention(self):
        """``_flex_head_dim`` sizes the flex block mask but is computed at construction,
        so it can drift from the head dimension attention actually sees. Check it against
        the real q/k/v; a mismatch resurfaces as a block-size crash on GPU."""
        for net in ({}, dict(hidden_v_channels=128, hidden_s_channels=512, num_heads=16)):
            with self.subTest(net=net or "small"):
                cfg = dict(input_dim=17, num_classes=10, trim=False, **{**_SMALL_NET, **net})
                torch.manual_seed(0)
                model = LGATrSlimTagger(**cfg)
                model.eval()

                seen = []
                attention = model.net.blocks[0].attention
                orig = attention._pre_attention_reshape

                def spy(*args, **kwargs):
                    q, k, v = orig(*args, **kwargs)
                    seen.append(q.shape[-1])
                    return q, k, v

                attention._pre_attention_reshape = spy
                with torch.no_grad():
                    model(*_make_inputs())
                self.assertEqual(seen[0], model._flex_head_dim)

    def test_flex_block_size_divides_inductor_config(self):
        """The tile must be a multiple of the BLOCK_M / BLOCK_N of the Triton config
        Inductor picks, or the lowering raises "Q and KV block size must be divisible by
        BLOCK_M and BLOCK_N". Only head dims 64 / 128 / 192 / 256 have tuned table
        entries, so those are where a fixed tile breaks."""
        try:
            from torch._inductor.virtualized import V
        except ImportError:
            self.skipTest("torch._inductor.virtualized.V is unavailable")

        device_type = "cuda" if torch.cuda.is_available() else "cpu"
        for head_dim in (16, 28, 64, 96, 128, 192, 256):
            with self.subTest(head_dim=head_dim):
                q_block, kv_block = _flex_block_size(head_dim, device_type, (torch.float32,))
                self.assertGreater(q_block, 0)
                self.assertGreater(kv_block, 0)
                try:
                    fwd = V.choices.get_flex_attention_fwd_configs(
                        head_dim, torch.float32, device_type
                    )
                    bwd = V.choices.get_flex_attention_bwd_configs(
                        head_dim, torch.float32, device_type
                    )
                except Exception:  # noqa: BLE001 -- private API; the fallback covers it
                    self.assertEqual((q_block, kv_block), (128, 128))
                    continue
                # at least one config of each must survive Inductor's divisibility filter
                self.assertTrue(any(q_block % c.block_m == 0 and kv_block % c.block_n == 0
                                    for c in fwd))
                self.assertTrue(
                    any(
                        q_block % c.block_m1 == 0
                        and q_block % c.block_m2 == 0
                        and kv_block % c.block_n1 == 0
                        and kv_block % c.block_n2 == 0
                        for c in bwd
                    )
                )

    def test_flex_requires_head_dim(self):
        """Sizing the block mask needs the head dimension, so it cannot be omitted."""
        ptr = torch.tensor([0, 3, 5], device="meta")
        batch = torch.tensor([0, 0, 0, 1, 1], device="meta")
        with self.assertRaises(ValueError):
            get_sparse_attention_kwargs(ptr, batch, maxlen=3, attention_backend="flex")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for flex_attention")
    def test_flex_backend_cuda_tuned_head_dim(self):
        """head_dim 64 hits a tuned row of Inductor's flex config table asking for
        BLOCK_M=128, wider than the 64-token tile that suits most head dims. This config
        used to fail to compile outright."""
        dense, packed = self._make_pair(
            "flex", hidden_v_channels=128, hidden_s_channels=512, num_heads=16
        )
        self.assertEqual(packed._flex_head_dim, 64)
        dense, packed = dense.cuda(), packed.cuda()
        x, v, mask = (t.cuda() for t in _make_inputs())
        with torch.no_grad():
            out_dense = dense(x, v, mask)
            out_packed = packed(x, v, mask)
        torch.testing.assert_close(out_dense, out_packed, rtol=1e-4, atol=1e-5)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for flex_attention")
    def test_flex_backend_cuda_varying_token_counts(self):
        """The compiled block-mask builder must keep the token count symbolic and the
        block size constant. If the block size is unspecialized too, the ceil-div in the
        mask's shape becomes a ratio of two unknowns and Inductor fails with "failed to
        set ranges" -- several steps in, not on the first, so this walks through a dozen
        training-shaped batches rather than testing one shape."""
        _, packed = self._make_pair("flex")
        packed = packed.cuda().train()
        g = torch.Generator().manual_seed(0)
        for step in range(12):
            seq_len = 24 + step
            x, v, mask = _make_inputs(batch=8, seq_len=seq_len, seed=step)
            lens = torch.randint(4, seq_len + 1, (8,), generator=g)
            mask = (torch.arange(seq_len)[None, :] < lens[:, None]).float().unsqueeze(1)
            out = packed(x.cuda(), v.cuda(), mask.cuda())
            out.sum().backward()
            packed.zero_grad(set_to_none=True)
            self.assertTrue(torch.isfinite(out).all())

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



_ETA = torch.diag(torch.tensor([1.0, -1.0, -1.0, -1.0], dtype=torch.float64))
_ETA_LIGHTCONE = torch.tensor(
    [[0.0, 1, 0, 0], [1, 0, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]], dtype=torch.float64
)


def _make_event_inputs(kind="quad", batch=3, num_features=17, seq_len=40, seed=0):
    """Multi-jet events (not a single jet): particle clouds around a few directions.

    Returns weaver-style ``(x, v, mask)`` in float64, with ``v`` ordered (px, py, pz, E).
    """
    generator = torch.Generator().manual_seed(seed)
    axes = {
        # back-to-back, so the summed momentum vanishes and the jet axis degenerates
        "dijet": [[0.3, 0.4, 0.86], [-0.3, -0.4, -0.86]],
        # exactly along the beam, where the z x n transverse pair degenerates
        "beam": [[0.0, 0, 1], [0, 0, -1]],
        "quad": [[1.0, 0, 0.2], [-1, 0.1, 0.3], [0, 1, -0.5], [0.1, 0.2, 1.0]],
        "isotropic": None,
    }[kind]
    momenta = []
    for _ in range(batch):
        if axes is None:
            directions = torch.randn(seq_len, 3, generator=generator, dtype=torch.float64)
        else:
            base = torch.tensor(axes, dtype=torch.float64)
            base = base[torch.arange(seq_len) % len(axes)]
            base = base / base.norm(dim=-1, keepdim=True)
            directions = base + 0.02 * torch.randn(
                seq_len, 3, generator=generator, dtype=torch.float64
            )
        directions = directions / directions.norm(dim=-1, keepdim=True)
        energy = torch.rand(seq_len, generator=generator, dtype=torch.float64) * 50 + 1
        mass = torch.rand(seq_len, generator=generator, dtype=torch.float64) * 0.3
        momenta.append(
            torch.cat([directions * energy[:, None], (energy**2 + mass**2).sqrt()[:, None]], -1)
        )
    v = torch.stack(momenta).transpose(1, 2)
    x = torch.randn(batch, num_features, seq_len, generator=generator, dtype=torch.float64)
    mask = torch.zeros(batch, 1, seq_len, dtype=torch.float64)
    for i in range(batch):
        mask[i, 0, : max(2, seq_len - 8 * i)] = 1
    return x, v, mask


def _event(directions, energies):
    """One massless event, ``(1, N, 4)`` in (E, px, py, pz), from unnormalized directions."""
    directions = torch.tensor(directions, dtype=torch.float64)
    directions = directions / directions.norm(dim=-1, keepdim=True)
    energies = torch.tensor(energies, dtype=torch.float64)
    momenta = torch.cat([energies[:, None], energies[:, None] * directions], dim=-1)
    return momenta.unsqueeze(0), torch.ones(1, len(energies), dtype=torch.bool)


_AXIS_FITS = {
    "principal": get_principal_axis,
    "collinear": lambda momenta, mask: get_collinear_axis(momenta, mask, scale=0.1),
}


class LightconeAxisGeometryTest(unittest.TestCase):
    """The frame construction and the particle-only axis fits of the full-event frames."""

    def test_frame_identities_for_every_direction(self):
        generator = torch.Generator().manual_seed(0)
        axes = torch.randn(512, 3, generator=generator, dtype=torch.float64)
        axes = axes / axes.norm(dim=-1, keepdim=True)
        # the directions that degenerate the z x n construction of get_lightcone_frame
        poles = torch.tensor(
            [[0.0, 0, 1], [0, 0, -1], [1, 0, 0], [0, 1, 0], [1e-18, 0, 1]], dtype=torch.float64
        )
        axes = torch.cat([axes, poles / poles.norm(dim=-1, keepdim=True)])
        frame = lightcone_frame_from_axis(axes)
        eye = torch.eye(4, dtype=torch.float64).expand_as(frame)
        torch.testing.assert_close(frame @ frame.transpose(-1, -2), eye)
        torch.testing.assert_close(
            frame.transpose(-1, -2) @ _ETA_LIGHTCONE @ frame, _ETA.expand_as(frame)
        )

    def test_frame_maps_momenta_to_lightcone_components(self):
        axis = torch.tensor([[0.0, 0.6, 0.8]], dtype=torch.float64)
        momentum = torch.tensor([[10.0, 1.0, 2.0, 3.0]], dtype=torch.float64)  # (E, px, py, pz)
        mapped = torch.einsum("nij,nj->ni", lightcone_frame_from_axis(axis), momentum)
        projection = (momentum[:, 1:] * axis).sum(-1)
        torch.testing.assert_close(mapped[:, 0], (momentum[:, 0] + projection) / 2**0.5)
        torch.testing.assert_close(mapped[:, 1], (momentum[:, 0] - projection) / 2**0.5)

    def test_leading_eigenvector_sign_and_degeneracy(self):
        axis = torch.tensor([0.0, -0.6, 0.8], dtype=torch.float64)
        matrix = torch.outer(axis, axis).expand(1, 3, 3)
        found = leading_eigenvector(matrix)[0]
        torch.testing.assert_close(found.abs(), axis.abs())
        # deterministic sign: the largest-magnitude component is positive
        self.assertGreater(found[found.abs().argmax()], 0)
        zero = leading_eigenvector(torch.zeros(1, 3, 3, dtype=torch.float64))
        torch.testing.assert_close(zero, torch.tensor([[0.0, 0, 1]], dtype=torch.float64))

    def test_axes_survive_a_balanced_dijet(self):
        # the momentum sum vanishes here, so the jet axis of get_lightcone_frame is noise
        _, v, mask = _make_event_inputs("dijet", batch=1, seq_len=40)
        fourmomenta = v.transpose(1, 2)[..., [3, 0, 1, 2]]
        expected = torch.tensor([0.3, 0.4, 0.86], dtype=torch.float64)
        expected = expected / expected.norm()
        for name, fit in _AXIS_FITS.items():
            with self.subTest(axis=name):
                axis = fit(fourmomenta, mask.squeeze(1).bool())[0]
                # one axis covers both poles, so either sign is correct
                self.assertGreater((axis * expected).sum().abs().item(), 0.99)

    def test_degenerate_events_give_valid_axes(self):
        cases = {
            "empty": (torch.ones(1, 6, 4, dtype=torch.float64), torch.zeros(1, 6, dtype=torch.bool)),
            "at_rest": (
                torch.tensor([[[1.0, 0, 0, 0]] * 6], dtype=torch.float64),
                torch.ones(1, 6, dtype=torch.bool),
            ),
            "single": _event([[1.0, 2.0, 3.0]], [5.0]),
            "duplicates": _event([[1.0, 1.0, 1.0]] * 12, [2.0] * 12),
            "isotropic": (
                torch.cat(
                    [
                        torch.ones(1, 200, 1, dtype=torch.float64),
                        torch.nn.functional.normalize(
                            torch.randn(1, 200, 3, generator=torch.Generator().manual_seed(1),
                                        dtype=torch.float64),
                            dim=-1,
                        ),
                    ],
                    dim=-1,
                ),
                torch.ones(1, 200, dtype=torch.bool),
            ),
        }
        for name, (fourmomenta, mask) in cases.items():
            for fit_name, fit in _AXIS_FITS.items():
                with self.subTest(case=name, axis=fit_name):
                    padded = torch.cat(
                        [fourmomenta, torch.full((1, 3, 4), float("nan"), dtype=torch.float64)], 1
                    )
                    padded_mask = torch.cat([mask, torch.zeros(1, 3, dtype=torch.bool)], 1)
                    axis = fit(padded, padded_mask)
                    self.assertTrue(torch.isfinite(axis).all())
                    torch.testing.assert_close(axis.norm(dim=-1), torch.ones(1, dtype=torch.float64))
                    frame = lightcone_frame_from_axis(axis)
                    torch.testing.assert_close(
                        frame.transpose(-1, -2) @ _ETA_LIGHTCONE @ frame, _ETA.expand_as(frame)
                    )

    def test_axes_are_permutation_invariant(self):
        _, v, mask = _make_event_inputs("quad", batch=2, seq_len=40)
        fourmomenta = v.transpose(1, 2)[..., [3, 0, 1, 2]]
        valid = mask.squeeze(1).bool()
        permutation = torch.randperm(40, generator=torch.Generator().manual_seed(3))
        for name, fit in _AXIS_FITS.items():
            with self.subTest(axis=name):
                # collinear sums its pairwise kernel in float32, in a permuted order
                torch.testing.assert_close(
                    fit(fourmomenta[:, permutation], valid[:, permutation]),
                    fit(fourmomenta, valid),
                    rtol=1e-5,
                    atol=1e-6,
                )


class CollinearAxisTest(unittest.TestCase):
    """``lightcone_frame="collinear"``: one frame, fitted to the conditioning objective.

    The axis is the leading eigenvector of ``sum_i c_i n_i n_i^T`` with ``c_i`` the
    nearby-momentum weight of :func:`collinearity_weights`, so the tests pin down what
    distinguishes it from ``principal``: it follows the collimated cluster rather than the
    isolated hard particle.
    """

    def test_follows_the_collimated_cluster_not_the_isolated_hard_particle(self):
        """A tight cluster along x outvotes one hard particle along z.

        `principal` weights by |p| and so points at the hard particle, whose pairs are all
        wide-angle and well conditioned; `collinear` weights by how many close neighbours
        a particle has and points at the cluster, where every internal pair is the one
        that actually loses precision in half precision.
        """
        cluster = [
            [1.0, 0.02 * i, 0.02 * j] for i in range(-2, 3) for j in range(-2, 3)
        ]  # 25 particles within ~0.06 rad of x
        # the single particle carries more momentum than the whole cluster
        momenta, mask = _event(cluster + [[0.0, 0.0, 1.0]], [1.0] * len(cluster) + [40.0])

        collinear = get_collinear_axis(momenta, mask, scale=0.1)[0]
        principal = get_principal_axis(momenta, mask)[0]
        self.assertGreater(collinear.abs()[0], 0.9, "collinear axis should follow the cluster (x)")
        self.assertGreater(principal.abs()[2], 0.9, "principal axis should follow the hard particle (z)")

    def test_weights_concentrate_on_dense_regions(self):
        momenta, mask = _event(
            [[1.0, 0.0, 0.0], [1.0, 0.01, 0.0], [1.0, 0.0, 0.01], [0.0, 0.0, 1.0]], [1.0] * 4
        )
        weights, directions = _weighted_directions(momenta, mask)
        c = collinearity_weights(directions, weights, weights > 0, 0.1)
        torch.testing.assert_close(c.sum(-1), torch.ones(1, dtype=torch.float64))
        self.assertGreater(c[0, 0].item(), 3 * c[0, 3].item())

    def test_self_pairs_are_excluded(self):
        """Without the j == i exclusion every particle would get ``w_i / scale^2`` on top,
        which turns ``c_i`` back into the momentum weight. Two isolated particles have no
        nearby partner, so their weights must follow the (tiny) cross-pair term alone,
        which is symmetric -- not their very different momenta."""
        momenta, mask = _event([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], [1.0, 9.0])
        weights, directions = _weighted_directions(momenta, mask)
        c = collinearity_weights(directions, weights, weights > 0, 0.01)
        torch.testing.assert_close(c, torch.full((1, 2), 0.5, dtype=torch.float64))

    def test_chunking_does_not_change_the_weights(self):
        torch.manual_seed(4)
        momenta = torch.rand(9, 16, 4, dtype=torch.float64) + 0.5
        momenta[..., 0] = momenta[..., 1:].norm(dim=-1)
        mask = torch.ones(9, 16, dtype=torch.bool)
        weights, directions = _weighted_directions(momenta, mask)
        whole = collinearity_weights(directions, weights, weights > 0, 0.1, chunk=64)
        chunked = collinearity_weights(directions, weights, weights > 0, 0.1, chunk=2)
        torch.testing.assert_close(whole, chunked)

    def test_collinear_weights_survive_an_outer_autocast(self):
        """The geometry must not be silently computed in the autocast dtype.

        `.to(torch.float32)` does not protect `bmm`: under an outer autocast it casts
        float32 operands down (float64 operands are left alone). In bfloat16 the spacing
        near cos = 1 is 2**-8, so `1 - cos` is unresolvable below ~0.09 rad and exactly
        the collinear pairs this weight is about collapse to theta = 0.
        """
        torch.manual_seed(0)
        momenta = torch.rand(2, 48, 4, dtype=torch.float64) + 0.1
        momenta[..., 0] = momenta[..., 1:].norm(dim=-1)
        mask = torch.ones(2, 48, dtype=torch.bool)

        reference = get_collinear_axis(momenta, mask, scale=0.02)
        for dtype in (torch.bfloat16, torch.float16):
            with self.subTest(autocast=dtype):
                with torch.autocast(device_type="cpu", dtype=dtype):
                    inside = get_collinear_axis(momenta, mask, scale=0.02)
                torch.testing.assert_close(inside, reference, rtol=0, atol=0)


class EventLightconeModelTest(unittest.TestCase):
    """The choice of light-cone axis is a change of coordinates, not a change of model.

    Every ``lightcone_frame`` must reproduce the Cartesian model in float64, outputs and
    gradients alike, on events where the jet axis is degenerate as well as on ones where
    it is merely poor.
    """

    _FRAMES = ("sum", "principal", "collinear")

    def _run(self, x, v, mask, loss_kind, **kwargs):
        cfg = dict(input_dim=x.shape[1], num_classes=10, trim=False, **_SMALL_NET)
        cfg.update(kwargs)
        torch.manual_seed(0)
        model = LGATrSlimTagger(**cfg).double()
        with torch.no_grad():
            # the "small" q/k/v init otherwise hides most of the attention
            for block in model.net.blocks:
                block.attention.linear_in.weight_v.mul_(10)
                block.attention.linear_in.linear_s.weight.mul_(10)
        model.eval()
        inputs = x.clone().requires_grad_(True)
        out = model(inputs, v, mask)
        if loss_kind == "probe":
            probe = torch.arange(out.numel(), dtype=torch.float64).reshape(out.shape) - 3.0
            loss = (out * probe).sum()
        else:
            loss = torch.nn.functional.cross_entropy(
                out, torch.zeros(out.shape[0], dtype=torch.long)
            )
        names = [n for n, p in model.named_parameters() if p.requires_grad]
        params = [p for p in model.parameters() if p.requires_grad]
        grads = torch.autograd.grad(loss, params + [inputs])
        return out.detach(), dict(zip(names + ["input"], grads))

    def _assert_matches(self, reference, candidate, msg):
        (out_ref, grads_ref), (out, grads) = reference, candidate
        torch.testing.assert_close(out, out_ref, rtol=1e-9, atol=1e-10, msg=msg)
        for name, want in grads_ref.items():
            torch.testing.assert_close(grads[name], want, rtol=1e-8, atol=1e-10, msg=f"{msg}: {name}")

    def test_matches_cartesian_fp64(self):
        for kind in ["dijet", "beam", "quad", "isotropic"]:
            x, v, mask = _make_event_inputs(kind)
            for loss_kind in ["probe", "cross_entropy"]:
                reference = self._run(x, v, mask, loss_kind, vector_coord="cartesian")
                for frame in self._FRAMES:
                    with self.subTest(kind=kind, loss=loss_kind, frame=frame):
                        candidate = self._run(
                            x, v, mask, loss_kind, vector_coord="lightcone", lightcone_frame=frame
                        )
                        self._assert_matches(reference, candidate, f"{kind}/{loss_kind}/{frame}")

    def test_matches_cartesian_with_mean_aggregation_and_checkpointing(self):
        x, v, mask = _make_event_inputs("quad")
        for extra in [dict(mean_aggregation=True), dict(checkpoint_blocks=True)]:
            with self.subTest(**extra):
                reference = self._run(x, v, mask, "probe", vector_coord="cartesian", **extra)
                candidate = self._run(
                    x, v, mask, "probe", vector_coord="lightcone", lightcone_frame="collinear",
                    **extra,
                )
                self._assert_matches(reference, candidate, str(extra))

    def test_packed_matches_dense(self):
        x, v, mask = _make_event_inputs("quad")
        for frame in self._FRAMES:
            with self.subTest(frame=frame):
                cfg = dict(vector_coord="lightcone", lightcone_frame=frame)
                dense = self._run(x, v, mask, "probe", attention_backend="native", **cfg)
                packed = self._run(x, v, mask, "probe", attention_backend="flex", **cfg)
                self._assert_matches(dense, packed, frame)

    def test_padding_may_hold_anything(self):
        x, v, mask = _make_event_inputs("quad")
        valid = mask.bool()
        poisoned_v = torch.where(valid, v, torch.full_like(v, float("nan")))
        poisoned_x = torch.where(valid, x, torch.full_like(x, 1e30))
        for frame in self._FRAMES:
            for backend in ["native", "flex"]:
                with self.subTest(frame=frame, backend=backend):
                    torch.manual_seed(0)
                    model = LGATrSlimTagger(
                        input_dim=x.shape[1], num_classes=10, trim=False, lightcone_frame=frame,
                        attention_backend=backend, **_SMALL_NET,
                    ).double().eval()
                    with torch.no_grad():
                        clean = model(x, v, mask)
                        poisoned = model(poisoned_x, poisoned_v, mask)
                    torch.testing.assert_close(poisoned, clean)

    def test_permutation_invariance(self):
        x, v, mask = _make_event_inputs("quad", batch=1, seq_len=24)
        permutation = torch.randperm(24, generator=torch.Generator().manual_seed(5))
        for frame in self._FRAMES:
            with self.subTest(frame=frame):
                torch.manual_seed(0)
                model = LGATrSlimTagger(
                    input_dim=x.shape[1], num_classes=10, trim=False, lightcone_frame=frame,
                    **_SMALL_NET,
                ).double().eval()
                with torch.no_grad():
                    out = model(x, v, mask)
                    out_permuted = model(
                        x[..., permutation], v[..., permutation], mask[..., permutation]
                    )
                torch.testing.assert_close(out_permuted, out, rtol=1e-9, atol=1e-10)

    def test_frame_follows_the_option(self):
        _, v, mask = _make_event_inputs("quad")
        fourmomenta = v.transpose(1, 2)[..., [3, 0, 1, 2]]
        valid = mask.squeeze(1).bool()
        expected = {
            "sum": get_lightcone_frame(fourmomenta, valid),
            "principal": lightcone_frame_from_axis(get_principal_axis(fourmomenta, valid)),
            "collinear": lightcone_frame_from_axis(get_collinear_axis(fourmomenta, valid)),
        }
        for frame, want in expected.items():
            with self.subTest(frame=frame):
                model = LGATrSlimTagger(
                    input_dim=17, num_classes=10, lightcone_frame=frame, **_SMALL_NET
                )
                torch.testing.assert_close(model._lightcone_frame(fourmomenta, valid), want)

    def test_rejects_unusable_configurations(self):
        with self.assertRaises(ValueError):  # no frame to choose in Cartesian coordinates
            LGATrSlimTagger(input_dim=4, num_classes=2, vector_coord="cartesian",
                            lightcone_frame="principal")
        with self.assertRaises(ValueError):
            LGATrSlimTagger(input_dim=4, num_classes=2, lightcone_frame="nonesuch")


# flex_attention and the flash-derived kernels need Ampere or newer; on older cards the
# Triton lowering itself refuses to compile, which is a property of the box, not the code
_FLEX_CAPABLE = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8


class EventLightconeCudaTest(unittest.TestCase):
    """GPU-only checks of the fitted-axis frames on real kernels."""

    def tearDown(self):
        # The eager flex path shares one ``torch.compile(dynamic=True)`` flex_attention
        # across calls; after the AMP dtypes (a different BlockMask tile) have gone
        # through it, a later float32 call in the same process can trip Inductor's
        # "failed to set ranges" -- start every other test from a clean cache.
        torch._dynamo.reset()

    @unittest.skipUnless(_FLEX_CAPABLE, "flex_attention needs an Ampere or newer GPU")
    def test_compiled_matches_eager_on_gpu(self):
        """``torch.compile`` must not change what the frame computation does.

        Training compiles while the equivalence tests above run eager, and the axis fits
        are exactly the kind of code Inductor can get wrong: ``collinear`` builds its
        weights with in-place arccos/square/reciprocal on an O(N^2) intermediate.
        """
        # flex_attention on a static toy shape only lowers for some head dims (24 and 32
        # raise NoValidChoicesError on sm_120 + torch 2.13), so use 8*4+32 = 64 in one head
        net = dict(hidden_v_channels=8, hidden_s_channels=32, num_blocks=2, num_heads=1)
        for frame in ("sum", "principal", "collinear"):
            with self.subTest(lightcone_frame=frame):
                torch.manual_seed(0)
                model = LGATrSlimTagger(
                    input_dim=17, num_classes=10, trim=False, lightcone_frame=frame,
                    attention_backend="flex", **net,
                ).cuda().eval()
                compiled = torch.compile(model)
                for seed in (0, 1):
                    # same shape, different content: a geometry baked in as a constant on
                    # the first call would only show up from the second batch on
                    x, v, mask = (t.float().cuda() for t in _make_event_inputs(seed=seed))
                    with torch.no_grad():
                        torch.testing.assert_close(
                            compiled(x, v, mask), model(x, v, mask), rtol=1e-4, atol=1e-5
                        )
                torch._dynamo.reset()

    @unittest.skipUnless(_FLEX_CAPABLE, "flex_attention needs an Ampere or newer GPU")
    def test_packed_backward_and_amp_on_gpu(self):
        x, v, mask = (t.float().cuda() for t in _make_event_inputs("quad", seq_len=48))
        for frame in ("principal", "collinear"):
            for amp_dtype in (None, torch.bfloat16, torch.float16):
                with self.subTest(frame=frame, amp=amp_dtype):
                    torch.manual_seed(0)
                    model = LGATrSlimTagger(
                        input_dim=x.shape[1], num_classes=10, trim=False, lightcone_frame=frame,
                        attention_backend="flex", **_SMALL_NET,
                    ).cuda()
                    model.train()
                    with torch.autocast("cuda", dtype=amp_dtype or torch.float32,
                                        enabled=amp_dtype is not None):
                        out = model(x, v, mask)
                    loss = torch.nn.functional.cross_entropy(
                        out.float(), torch.zeros(out.shape[0], dtype=torch.long, device="cuda")
                    )
                    loss.backward()
                    grads = [p.grad for p in model.parameters() if p.requires_grad]
                    self.assertTrue(all(g is not None for g in grads))
                    self.assertTrue(all(torch.isfinite(g).all() for g in grads))


class AmpPolicyTest(unittest.TestCase):
    """Precision is the trainer's business, not the model's.

    weaver enters ``torch.autocast`` once per step in ``weaver.utils.nn.tools`` and passes
    ``use_amp=True`` to the model as well (``train.py``). The model must not act on that
    second copy: a model-side ``torch.autocast(enabled=self.use_amp)`` *overrides* the
    surrounding region, so with ``use_amp=False`` -- the default, and what every script
    that builds the tagger directly gets -- it silently forced the whole forward back to
    float32 while the caller believed it was measuring AMP.
    """

    def _model(self, **kwargs):
        torch.manual_seed(1)
        return LGATrSlimTagger(
            input_dim=17, num_classes=10, trim=False, **_SMALL_NET, **kwargs
        ).eval()

    def test_use_amp_does_not_change_the_forward(self):
        """Same weights, same inputs, both settings: bitwise equal, in or out of autocast."""
        x, v, mask = _make_inputs()
        off, on = self._model(use_amp=False), self._model(use_amp=True)
        with torch.no_grad():
            self.assertTrue(torch.equal(off(x, v, mask), on(x, v, mask)))
            with torch.autocast("cpu", dtype=torch.bfloat16):
                self.assertTrue(torch.equal(off(x, v, mask), on(x, v, mask)))

    def test_use_amp_does_not_enable_autocast_by_itself(self):
        x, v, mask = _make_inputs()
        with torch.no_grad():
            self.assertEqual(self._model(use_amp=True)(x, v, mask).dtype, torch.float32)

    def test_the_trainer_s_autocast_reaches_the_network(self):
        """The regression guard: ``use_amp=False`` must not veto an outer AMP region."""
        x, v, mask = _make_inputs()
        with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
            self.assertEqual(self._model(use_amp=False)(x, v, mask).dtype, torch.bfloat16)

    def test_the_model_opens_no_autocast_region_of_its_own(self):
        """Nothing inside the forward may *enable* autocast; the fp32 islands only disable it."""
        x, v, mask = _make_inputs()
        model = self._model(use_amp=True)
        real = torch.autocast
        enabled_calls = []

        class _spy(real):
            def __init__(self, *args, **kwargs):
                if kwargs.get("enabled", True) and (len(args) < 2 or args[1]):
                    enabled_calls.append((args, kwargs))
                super().__init__(*args, **kwargs)

        torch.autocast = _spy
        try:
            with torch.no_grad():
                model(x, v, mask)
        finally:
            torch.autocast = real
        self.assertEqual(enabled_calls, [])


if __name__ == "__main__":
    unittest.main()
