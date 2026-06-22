"""Unit tests for exporting ParticleTransformer to ONNX.

These drive the real export entry point, ``weaver.train.onnx``, so the whole
production path (``model_setup`` -> checkpoint load -> ``torch.onnx.export`` ->
``preprocess.json``) is covered. They guard in particular:

* ``version >= 3`` uses RMS normalization. The native ``torch.nn.RMSNorm``
  cannot be exported by the TorchScript ONNX exporter (no ``aten::rms_norm``
  symbolic), so the model swaps in an ONNX-friendly ``RMSNorm`` when built with
  ``for_inference=True`` while keeping the native (fused) one for training.
* The exported graph must reproduce the PyTorch model's output numerically.
"""

import argparse
import os
import tempfile
import unittest

import numpy as np
import torch
import torch.nn as nn

try:
    import onnxruntime as ort

    _HAS_ORT = True
except ImportError:
    _HAS_ORT = False

from weaver import train as weaver_train
from weaver.utils.dataset import DataConfig
from weaver.utils.import_tools import import_module
from weaver.nn.model.ParticleTransformer import RMSNorm

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA_CONFIG = os.path.join(_HERE, "data", "JetClass_full.yaml")
_NETWORK_CONFIG = os.path.join(_HERE, "networks", "example_ParT.py")
_NATIVE_RMSNORM = getattr(nn, "RMSNorm", None)


def _make_args(workdir, version):
    """Minimal args namespace accepted by ``weaver.train.onnx`` / ``model_setup``."""
    return argparse.Namespace(
        data_config=_DATA_CONFIG,
        network_config=_NETWORK_CONFIG,
        network_option=[["version", str(version)]],
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


def _physical_inputs(data_config, batch=1, seq_len=None, seed=1):
    """Realistic inputs: non-zero (wrap-like) 4-vectors to avoid atan2(0, 0) NaNs.

    `batch` and `seq_len` may differ from the configured shapes to exercise the
    exported model's dynamic axes.
    """
    g = torch.Generator().manual_seed(seed)
    if seq_len is None:
        seq_len = data_config.input_shapes["pf_mask"][2]
    shapes = {k: (batch, s[1], seq_len) for k, s in data_config.input_shapes.items()}
    inp = {k: torch.randn(*shp, generator=g) for k, shp in shapes.items()}
    mask = torch.zeros(batch, 1, seq_len)
    for i in range(batch):
        mask[i, 0, : min(seq_len, 4 + 3 * i)] = 1
    inp["pf_mask"] = mask
    px, py, pz = (torch.randn(batch, 1, seq_len, generator=g) + 0.5 for _ in range(3))
    m = torch.rand(batch, 1, seq_len, generator=g) * 0.3 + 0.1
    inp["pf_vectors"] = torch.cat([px, py, pz, torch.sqrt(px**2 + py**2 + pz**2 + m**2)], dim=1)
    return inp


@unittest.skipUnless(_HAS_ORT, "onnxruntime is required for ONNX export tests")
class OnnxExportTest(unittest.TestCase):
    def _check_version(self, version):
        data_config = DataConfig.load(_DATA_CONFIG, load_observers=False, load_reweight_info=False)
        net = import_module(_NETWORK_CONFIG, name="_onnx_test_net")

        with tempfile.TemporaryDirectory() as workdir:
            args = _make_args(workdir, version)

            # build a training-mode model (uses native nn.RMSNorm for v3) and
            # save it as the checkpoint that `train.onnx` will load and export
            train_model, _ = net.get_model(data_config, version=version)
            train_model.eval()
            torch.save(train_model.state_dict(), args.model_prefix)

            # === the export path under test ===
            weaver_train.onnx(args)

            self.assertTrue(os.path.isfile(args.export_onnx))
            self.assertTrue(os.path.isfile(os.path.join(workdir, "preprocess.json")))

            sess = ort.InferenceSession(args.export_onnx, providers=["CPUExecutionProvider"])
            seq_len = data_config.input_shapes["pf_mask"][2]
            # exercise the declared-dynamic axes: vary both batch size and
            # sequence length away from the configured (1, seq_len) export shape
            for batch, plen in [(1, seq_len), (4, seq_len), (1, seq_len // 2), (3, seq_len // 2)]:
                with self.subTest(version=version, batch=batch, seq_len=plen):
                    inp = _physical_inputs(data_config, batch=batch, seq_len=plen)
                    with torch.no_grad():
                        logits = train_model(*[inp[k] for k in data_config.input_names])
                        ref = torch.softmax(logits, dim=1).numpy()
                    feed = {i.name: inp[i.name].numpy().astype(np.float32) for i in sess.get_inputs()}
                    out = sess.run(None, feed)[0]
                    self.assertFalse(np.isnan(out).any(), msg=f"v{version}: ONNX output contains NaNs")
                    np.testing.assert_allclose(
                        out, ref, rtol=1e-3, atol=1e-4, err_msg=f"v{version}: ONNX output differs from PyTorch"
                    )

    def test_export_v1(self):
        self._check_version(1)

    def test_export_v2(self):
        self._check_version(2)

    def test_export_v3(self):
        self._check_version(3)

    @unittest.skipUnless(_NATIVE_RMSNORM is not None, "torch.nn.RMSNorm is unavailable")
    def test_v3_rmsnorm_implementation_depends_on_for_inference(self):
        data_config = DataConfig.load(_DATA_CONFIG, load_observers=False, load_reweight_info=False)
        net = import_module(_NETWORK_CONFIG, name="_onnx_test_net")

        # training keeps the native (fused) RMSNorm; export swaps in the
        # ONNX-exportable implementation
        train_mods = list(net.get_model(data_config, version=3)[0].modules())
        export_mods = list(net.get_model(data_config, version=3, for_inference=True)[0].modules())

        self.assertTrue(any(isinstance(m, _NATIVE_RMSNORM) for m in train_mods))
        self.assertFalse(any(type(m) is RMSNorm for m in train_mods))

        self.assertTrue(any(type(m) is RMSNorm for m in export_mods))
        self.assertFalse(any(isinstance(m, _NATIVE_RMSNORM) for m in export_mods))


if __name__ == "__main__":
    unittest.main()
