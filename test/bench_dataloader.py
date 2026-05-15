#!/usr/bin/env python3
"""Benchmark script for weaver-core dataloader."""

import argparse
import glob
import logging
import multiprocessing
import time

# Python 3.14 defaults to forkserver which requires pickling the dataset;
# use fork to avoid that overhead and match typical training setups.
multiprocessing.set_start_method("fork", force=True)

logging.disable(logging.CRITICAL)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from weaver.utils.dataset import SimpleIterDataset


def build_loader(
    file_dict,
    data_config,
    *,
    for_training,
    fetch_by_files,
    fetch_step,
    batch_size,
    num_workers,
    prefetch_factor,
):
    ds = SimpleIterDataset(
        file_dict,
        data_config,
        batch_size=batch_size,
        for_training=for_training,
        load_range_and_fraction=((0, 1), 1.0, 1),
        fetch_by_files=fetch_by_files,
        fetch_step=fetch_step,
        name="train" if for_training else "test",
    )
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        drop_last=for_training,
        pin_memory=True,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
    )
    return loader


def run_one(loader, max_batches=None):
    t_prev = time.perf_counter()
    batch_times = []
    n_samples = 0
    for X, y, Z in loader:
        now = time.perf_counter()
        batch_times.append(now - t_prev)
        t_prev = now
        n_samples += list(y.values())[0].shape[0]
        if max_batches and len(batch_times) >= max_batches:
            break
    return len(batch_times), n_samples, batch_times


def parse_list(s, typ):
    return [typ(x) for x in s.split(",")]


def main():
    parser = argparse.ArgumentParser(description="Dataloader benchmark")
    parser.add_argument("--data-config", type=str, default="test/data/JetClass_full.yaml")
    parser.add_argument("--data-dir", type=str, default="/data/hqu/datasets/JetClass/Pythia/val_5M")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--max-batches", type=int, default=200,
                        help="Max batches per benchmark run (0 = exhaust loader)")
    parser.add_argument("--mode", choices=["train", "test", "both"], default="both",
                        help="Run train, test, or both (default: both)")
    parser.add_argument("--batch-size", type=str, default="1024",
                        help="Comma-separated batch sizes (default: 1024)")
    parser.add_argument("--fetch-step", type=str, default="0.05",
                        help="Comma-separated fetch steps for train mode (default: 0.05)")
    parser.add_argument("--fetch-by-files", action="store_true", default=None,
                        help="Use fetch_by_files=True in train mode (default: False)")
    args = parser.parse_args()

    files = sorted(glob.glob(f"{args.data_dir}/*.root")) + sorted(glob.glob(f"{args.data_dir}/*.parquet"))
    if not files:
        raise RuntimeError(f"No .root or .parquet files found in {args.data_dir}")
    file_dict = {"_": files}
    max_batches = args.max_batches or None

    batch_sizes = parse_list(args.batch_size, int)
    fetch_steps = parse_list(args.fetch_step, float)

    run_train = args.mode in ("train", "both")
    run_test = args.mode in ("test", "both")

    print(f"Files: {len(files)}  |  Workers: {args.num_workers}  |  Max batches: {max_batches}")
    print()

    scenarios = []

    if run_train:
        fetch_by_files = bool(args.fetch_by_files)
        for fetch_step in fetch_steps:
            for batch_size in batch_sizes:
                scenarios.append(
                    dict(
                        label=f"train  fetch_step={fetch_step:<5}  bs={batch_size}"
                              + (f"  fetch_by_files" if fetch_by_files else ""),
                        for_training=True,
                        fetch_by_files=fetch_by_files,
                        fetch_step=fetch_step,
                        batch_size=batch_size,
                    )
                )

    if run_test:
        for batch_size in batch_sizes:
            scenarios.append(
                dict(
                    label=f"test   fetch_by_files=True  fetch_step=1  bs={batch_size}",
                    for_training=False,
                    fetch_by_files=True,
                    fetch_step=1,
                    batch_size=batch_size,
                )
            )

    header = f"{'scenario':<58} {'batches':>8} {'samples':>10} {'1st(s)':>8} {'rest(s)':>8} {'total(s)':>8} {'samp/s':>10}"
    print(header)
    print("-" * len(header))

    all_results = []
    for s in scenarios:
        label = s.pop("label")
        loader = build_loader(
            file_dict,
            args.data_config,
            num_workers=args.num_workers,
            prefetch_factor=args.prefetch_factor,
            **s,
        )
        n_batches, n_samples, batch_times = run_one(loader, max_batches)
        total = sum(batch_times)
        t_first = batch_times[0] if batch_times else 0
        t_rest = total - t_first
        rate = n_samples / total if total > 0 else float("inf")
        print(f"{label:<58} {n_batches:>8d} {n_samples:>10d} {t_first:>8.2f} {t_rest:>8.2f} {total:>8.2f} {rate:>10.0f}")
        all_results.append((label, batch_times))

    # per-batch timing details
    print("\n\n=== Per-batch timing details ===")
    for label, batch_times in all_results:
        if not batch_times:
            continue
        bt = np.array(batch_times)
        # classify batches: "load" batches (stalls waiting for new data) vs "fast" batches (serving from buffer)
        if len(bt) > 1:
            median = np.median(bt[1:])
            threshold = max(median * 5, 0.5)
            load_mask = bt > threshold
            load_mask[0] = True  # first batch is always a load
            n_load = load_mask.sum()
            n_fast = len(bt) - n_load
            t_load = bt[load_mask].sum()
            t_fast = bt[~load_mask].sum() if n_fast > 0 else 0
        else:
            n_load, n_fast = 1, 0
            t_load, t_fast = bt[0], 0
            threshold = 0

        print(f"\n  {label}")
        print(f"    Total: {len(bt)} batches, {sum(bt):.2f}s")
        print(f"    Load batches (>{threshold:.3f}s): {n_load:>4d}  total {t_load:.2f}s  avg {t_load/max(n_load,1):.3f}s")
        print(f"    Fast batches:               {n_fast:>4d}  total {t_fast:.2f}s  avg {t_fast/max(n_fast,1):.4f}s")
        pcts = [0, 25, 50, 75, 90, 95, 99, 100]
        vals = np.percentile(bt, pcts)
        pct_str = "  ".join(f"p{p}={v:.4f}" for p, v in zip(pcts, vals))
        print(f"    Percentiles (s): {pct_str}")

    print("\nDone.")


if __name__ == "__main__":
    main()
