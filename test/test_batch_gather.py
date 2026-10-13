"""``batch_gather`` must yield exactly the batches of the per-sample path.

The dataset yields whole batches gathered from its table instead of single events for the
DataLoader to collate. Same events, same order, same chunk-spanning batches, same types and
dtypes -- for padded storage and every ``--jagged`` mode, shuffled training (several chunks
per worker, in-memory re-shuffles) and unshuffled testing (short final batch kept).

Run: ``python -m pytest test/test_batch_gather.py``.
"""

import os
import tempfile
import unittest

import awkward as ak
import numpy as np
import torch
from torch.utils.data import DataLoader

from weaver.utils.data.jagged import JaggedCollator, Unpadded
from weaver.utils.dataset import SimpleIterDataset

CONFIG = """
selection: null
new_variables:
   part_mask: ak.ones_like(part_x)
   part_x2: part_x * 2
inputs:
   pf_features:
      length: 16
      pad_mode: wrap
      vars:
         - [part_x, 0.5, 2.0]
         - [part_y, null]
   pf_extra:
      length: 16
      pad_mode: constant
      vars:
         - [part_x2, null, 1, -5, 5, 0.25]
   pf_mask:
      length: 16
      pad_mode: constant
      vars:
         - [part_mask, null]
labels:
   type: simple
   value: [label_a, label_b]
observers:
   - evt
   - nconst
   - part_y
weights: null
"""


def _write_files(directory, n_files=4, n_events=700):
    rng = np.random.default_rng(0)
    paths = []
    for f in range(n_files):
        counts = rng.integers(0, 24, n_events)  # empty events and events longer than 16
        flat = lambda: rng.normal(0, 1, counts.sum()).astype(np.float32)  # noqa: E731
        label = rng.integers(0, 2, n_events)
        table = ak.Array({
            "part_x": ak.unflatten(flat(), counts),
            "part_y": ak.unflatten(flat(), counts),
            "label_a": (label == 0).astype(np.int32),
            "label_b": (label == 1).astype(np.int32),
            "evt": np.arange(n_events, dtype=np.int64) + 10000 * f,
            "nconst": counts.astype(np.float32),
        })
        path = os.path.join(directory, f"f{f}.parquet")
        ak.to_parquet(table, path, row_group_size=100)
        paths.append(path)
    return paths


class BatchGatherTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.files = _write_files(cls.tmp.name)
        cls.config = os.path.join(cls.tmp.name, "cfg.yaml")
        with open(cls.config, "w") as f:
            f.write(CONFIG)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def _batches(self, batch_gather, jagged, for_training, in_memory=False, workers=2, bs=96, n=None):
        # workers reseed numpy from the DataLoader generator; in-process loading uses the global state
        np.random.seed(0)
        ds = SimpleIterDataset(
            {"a": self.files[:2], "b": self.files[2:]}, self.config, batch_size=bs,
            for_training=for_training, load_range_and_fraction=((0, 1), 1, 1),
            fetch_by_files=not for_training, fetch_step=0.3 if for_training else 1,
            infinity_mode=in_memory, in_memory=in_memory, jagged=jagged or False,
            batch_gather=batch_gather, drop_last=for_training, name="train" if for_training else "test",
        )
        if batch_gather:
            kw = dict(batch_size=None)
        else:
            kw = dict(batch_size=bs, drop_last=for_training,
                      collate_fn=JaggedCollator(ds.config, jagged) if jagged else None)
        dl = DataLoader(ds, num_workers=workers, generator=torch.Generator().manual_seed(7), **kw)
        out = []
        for i, b in enumerate(dl):
            out.append(b)
            if n is not None and i + 1 == n:
                break
        return out

    def _assert_same(self, ref, got, msg):
        self.assertEqual(len(ref), len(got), msg)
        for bi, (r, g) in enumerate(zip(ref, got)):
            for part, (rd, gd) in enumerate(zip(r, g)):
                self.assertEqual(list(rd), list(gd), msg)
                for k in rd:
                    a, b = rd[k], gd[k]
                    where = f"{msg}: batch {bi} part {part} key {k}"
                    if isinstance(a, Unpadded):
                        self.assertIsInstance(b, Unpadded, where)
                        self.assertTrue(all(torch.equal(p, q) for p, q in zip(a, b)), where)
                    elif isinstance(a, torch.Tensor) and a.is_nested:
                        self.assertTrue(b.is_nested, where)
                        self.assertTrue(torch.equal(a.values(), b.values()), where)
                        self.assertTrue(torch.equal(a.offsets(), b.offsets()), where)
                    elif isinstance(a, torch.Tensor):
                        self.assertIsInstance(b, torch.Tensor, where)
                        self.assertEqual(a.dtype, b.dtype, where)
                        self.assertTrue(torch.equal(a, b), where)
                    else:
                        self.assertEqual(type(a), type(b), where)
                        self.assertEqual(ak.to_list(a), ak.to_list(b), where)

    def test_training_batches(self):
        for jagged in (None, "host", "device", "nested"):
            with self.subTest(jagged=jagged):
                ref = self._batches(False, jagged, True)
                got = self._batches(True, jagged, True)
                # several chunks per worker, so some batches span two of them
                self.assertGreater(len(ref), 10)
                self._assert_same(ref, got, f"train jagged={jagged}")

    def test_device_padding_reproduces_stock_batches(self):
        """``--jagged device``: padding the gathered unpadded batch on the device gives the
        stock loader's batch, wrap and constant (non-zero pad value) groups alike."""
        stock = self._batches(False, None, True)
        device = self._batches(True, "device", True)
        self.assertEqual(len(stock), len(device))
        cfg = SimpleIterDataset({}, self.config, for_training=True).config
        padder = JaggedCollator(cfg, "device")
        for bi, ((xs, ys, _), (xd, yd, _)) in enumerate(zip(stock, device)):
            self.assertTrue(torch.equal(ys["_label_"], yd["_label_"]))
            for k in xs:
                self.assertTrue(torch.equal(xs[k], padder.pad_on_device(k, xd[k])), (bi, k))

    def test_in_memory_reshuffle(self):
        # 2 workers x ~1400 events in memory: 40 batches of 96 cross several re-shuffles
        for jagged in (None, "nested"):
            with self.subTest(jagged=jagged):
                ref = self._batches(False, jagged, True, in_memory=True, n=40)
                got = self._batches(True, jagged, True, in_memory=True, n=40)
                self._assert_same(ref, got, f"in-memory jagged={jagged}")

    def test_testing_batches_keep_the_short_tail(self):
        for jagged in (None, "host"):
            with self.subTest(jagged=jagged):
                ref = self._batches(False, jagged, False, bs=96)
                got = self._batches(True, jagged, False, bs=96)
                self.assertNotEqual(ref[-1][1]["_label_"].shape[0], 96)  # 1400 events per worker
                self._assert_same(ref, got, f"test jagged={jagged}")
                # observers: scalars become tensors, the jagged one stays an awkward array
                X, y, Z = got[0]
                self.assertEqual(sorted(Z), ["evt", "nconst", "part_y"])
                self.assertIsInstance(Z["part_y"], ak.Array)

    def test_single_process(self):
        ref = self._batches(False, None, True, workers=0)
        got = self._batches(True, None, True, workers=0)
        self._assert_same(ref, got, "num_workers=0")


if __name__ == "__main__":
    unittest.main()
