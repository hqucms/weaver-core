import random
import unittest

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset

from weaver.train import data_generator, parser, set_seed


def _args(seed=None):
    argv = ["--data-config", "dummy", "--network-config", "dummy"]
    if seed is not None:
        argv += ["--seed", str(seed)]
    args = parser.parse_args(argv)
    args.local_rank = None
    return args


class _SeedReportingDataset(IterableDataset):
    """Reports the worker seed and a draw from the numpy state SimpleIterDataset sets up."""

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        np.random.seed(info.seed & 0xFFFFFFFF)
        for _ in range(3):
            yield torch.tensor([info.seed % 100000, int(np.random.randint(0, 10**6))])


def _stream(seed, init_draws):
    args = _args(seed)
    set_seed(args)
    # stand-in for a model init: consumes a config-dependent amount of the global torch RNG
    torch.randn(init_draws)
    loader = DataLoader(
        _SeedReportingDataset(), num_workers=2, batch_size=3, generator=data_generator(args, "train")
    )
    return [batch.tolist() for batch in loader]


class SeedTest(unittest.TestCase):
    def test_set_seed_is_a_noop_without_seed(self):
        args = _args()
        self.assertIsNone(args.seed)
        set_seed(args)  # must not raise
        self.assertIsNone(data_generator(args, "train"))

    def test_set_seed_makes_the_main_process_rngs_reproducible(self):
        set_seed(_args(1234))
        drawn = (random.random(), float(np.random.rand()), float(torch.randn(1)))
        set_seed(_args(1234))
        self.assertEqual(drawn, (random.random(), float(np.random.rand()), float(torch.randn(1))))

    def test_data_generator_is_stable_per_loader_and_distinct_across_loaders(self):
        args = _args(7)
        self.assertEqual(data_generator(args, "train").initial_seed(), data_generator(args, "train").initial_seed())
        self.assertNotEqual(data_generator(args, "train").initial_seed(), data_generator(args, "val").initial_seed())

    def test_data_generator_differs_across_ranks(self):
        args = _args(7)
        args.local_rank = 0
        rank0 = data_generator(args, "train").initial_seed()
        args.local_rank = 1
        self.assertNotEqual(rank0, data_generator(args, "train").initial_seed())

    def test_worker_seeds_are_reproducible(self):
        self.assertEqual(_stream(42, 10), _stream(42, 10))

    def test_worker_seeds_do_not_depend_on_model_rng_consumption(self):
        """The point of the dedicated generator: two configs must see the same events."""
        self.assertEqual(_stream(42, 10), _stream(42, 9999))

    def test_different_seeds_give_different_data_order(self):
        self.assertNotEqual(_stream(42, 10), _stream(43, 10))


if __name__ == "__main__":
    unittest.main()
