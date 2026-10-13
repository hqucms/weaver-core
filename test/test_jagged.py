"""Tests for the unpadded (``--jagged``) loader path and LGATr-slim's jagged forward.

``--jagged host`` must reproduce the stock loader's batches bit for bit, ``device`` must do
so after padding on the device, and ``nested`` must hold exactly the real tokens. ``LGATrSlimTagger.forward_jagged`` must equal the padded forward.

Run: ``python -m pytest test/test_jagged.py``.
"""

import os
import unittest

import awkward as ak
import numpy as np
import torch
from torch.utils.data._utils.collate import default_collate

from weaver.utils.data.config import DataConfig
from weaver.utils.data.jagged import JaggedArray, JaggedCollator, Unpadded
from weaver.utils.dataset import _finalize_inputs

CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "JetClass_full.yaml")


def _table(cfg, rng, counts):
    """A synthetic post-``new_variables`` table: every input variable, jagged, plus labels.
    Includes NaNs and values outside the clip range, and rows longer than the pad length."""
    table = {}
    for names in cfg.input_dicts.values():
        for n in names:
            flat = rng.normal(0, 4, counts.sum()).astype(np.float32)
            flat[rng.random(len(flat)) < 0.01] = np.nan
            table[n] = ak.unflatten(ak.Array(flat), counts)
    table["_label_"] = rng.integers(0, 10, len(counts))
    return table


class TestJaggedCollate(unittest.TestCase):
    def setUp(self):
        self.cfg = DataConfig.load(CONFIG, load_observers=False)
        rng = np.random.default_rng(0)
        counts = rng.integers(0, 140, 300)  # includes empty rows and rows > 128
        counts[:3] = [0, 128, 139]
        self.stock = _finalize_inputs(_table(self.cfg, np.random.default_rng(1), counts), self.cfg)
        self.jagged = _finalize_inputs(_table(self.cfg, np.random.default_rng(1), counts), self.cfg, jagged=True)
        self.counts = np.minimum(counts, 128)
        self.idx = rng.permutation(len(counts))[:64]
        self.idx[:3] = [0, 1, 2]

    def _batch(self, out, collate):
        samples = [
            ({k: out["_" + k][i] for k in self.cfg.input_names}, {"_label_": out["_label_"][i]}, {})
            for i in self.idx
        ]
        return collate(samples)

    def test_storage_is_unpadded(self):
        for k in self.cfg.input_names:
            a = self.jagged["_" + k]
            self.assertIsInstance(a, JaggedArray)
            self.assertEqual(a.values.shape[0], self.counts.sum())
            self.assertLess(a.nbytes, self.stock["_" + k].nbytes)

    def test_host_matches_stock(self):
        ref = self._batch(self.stock, default_collate)
        got = self._batch(self.jagged, JaggedCollator(self.cfg, "host"))
        for k in self.cfg.input_names:
            self.assertTrue(torch.equal(ref[0][k], got[0][k]), k)
        self.assertTrue(torch.equal(ref[1]["_label_"], got[1]["_label_"]))

    def test_device_matches_stock_after_padding(self):
        # wrap- and constant-padded groups, empty events and events cut to the pad length
        ref = self._batch(self.stock, default_collate)
        collator = JaggedCollator(self.cfg, "device")
        got = self._batch(self.jagged, collator)
        for k in self.cfg.input_names:
            self.assertIsInstance(got[0][k], Unpadded, k)
            padded = collator.pad_on_device(k, got[0][k])
            self.assertEqual(padded.dtype, ref[0][k].dtype)
            self.assertTrue(padded.is_contiguous())
            self.assertTrue(torch.equal(ref[0][k], padded), k)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA device")
    def test_device_padding_on_cuda(self):
        ref = self._batch(self.stock, default_collate)
        collator = JaggedCollator(self.cfg, "device")
        got = self._batch(self.jagged, collator)
        for k in self.cfg.input_names:
            padded = collator.pad_on_device(k, got[0][k].to("cuda"))
            self.assertTrue(torch.equal(ref[0][k], padded.cpu()), k)

    def test_device_padding_of_an_all_empty_batch(self):
        collator = JaggedCollator(self.cfg, "device")
        for k in self.cfg.input_names:
            n_vars = len(self.cfg.input_dicts[k])
            empty = Unpadded(torch.zeros(0, n_vars), torch.zeros(4, dtype=torch.long))
            padded = collator.pad_on_device(k, empty)
            length, wrap, pad_values = collator.pad[k]
            want = torch.zeros(3, n_vars, length)
            if not wrap:
                want += torch.as_tensor(pad_values)[None, :, None]
            self.assertTrue(torch.equal(padded, want), k)

    def test_nested_holds_real_tokens(self):
        ref = self._batch(self.stock, default_collate)
        got = self._batch(self.jagged, JaggedCollator(self.cfg, "nested"))
        lengths = torch.as_tensor(self.counts[self.idx])
        for k in self.cfg.input_names:
            nt = got[0][k]
            self.assertTrue(nt.is_nested)
            self.assertTrue(torch.equal(nt.offsets().diff(), lengths))
            for b, n in enumerate(lengths.tolist()):
                self.assertTrue(torch.equal(nt[b], ref[0][k][b, :, :n].T), (k, b))


class TestLGATrSlimJagged(unittest.TestCase):
    def test_forward_jagged_matches_padded(self):
        from weaver.nn.model.LGATrSlim import LGATrSlimTagger

        torch.manual_seed(0)
        B, C, P = 12, 5, 40
        n = torch.randint(1, P, (B,))
        off = torch.cat([torch.zeros(1, dtype=torch.long), n.cumsum(0)])
        T = int(off[-1])
        xv = torch.randn(T, C)
        p3 = torch.randn(T, 3) + torch.randn(B, 3).repeat_interleave(n, 0) * 20
        vv = torch.cat([p3, (p3.square().sum(-1, keepdim=True) + 0.1).sqrt()], -1)
        mask = torch.arange(P)[None] < n[:, None]
        x, v = torch.zeros(B, P, C), torch.zeros(B, P, 4)
        x[mask], v[mask] = xv, vv
        for frame, mean in (("sum", False), ("principal", True)):
            net = LGATrSlimTagger(
                input_dim=C, num_classes=3, hidden_v_channels=4, hidden_s_channels=8, num_blocks=2,
                num_heads=2, attention_backend="varlen", lightcone_frame=frame, auxiliary_scalars="all",
                mean_aggregation=mean, trim=False,
            )
            dense = net(x.transpose(1, 2), v.transpose(1, 2), mask[:, None].float())
            packed = net.forward_jagged(xv, vv, off, maxlen=P)
            torch.testing.assert_close(packed, dense, rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
