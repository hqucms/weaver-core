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
from weaver.nn.model.LGATrSlim import LGATrSlimTagger

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


@unittest.skipUnless(_HAS_ORT, "onnxruntime is required for ONNX export tests")
class LGATrSlimOnnxExportTest(unittest.TestCase):
    def test_export(self):
        data_config = DataConfig.load(_DATA_CONFIG, load_observers=False, load_reweight_info=False)
        net = import_module(_NETWORK_CONFIG, name="_lgatr_onnx_test_net")

        with tempfile.TemporaryDirectory() as workdir:
            args = argparse.Namespace(
                data_config=_DATA_CONFIG,
                network_config=_NETWORK_CONFIG,
                network_option=[[k, str(val)] for k, val in _SMALL_NET.items()],
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
