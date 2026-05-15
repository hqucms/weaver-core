#!/usr/bin/env python3
"""Convert ROOT (or other supported formats) to Parquet with configurable row group size.

Parquet with small row groups enables efficient partial reads via pyarrow,
which is 6-12x faster than uproot for the same data.

Usage:
    python -m weaver.utils.convert_to_parquet /data/input_dir/*.root -o /data/output_dir/
    python -m weaver.utils.convert_to_parquet filelist.txt -o /data/output_dir/ --data-config config.yaml
    python -m weaver.utils.convert_to_parquet /data/input_dir/*.root -o /data/output_dir/ --compression zstd --row-group-size 5000
"""

import argparse
import glob
import os
import time

import awkward as ak
import tqdm


def convert_file(input_path, output_path, *, branches=None, treename=None, branch_magic=None,
                 compression="lz4", compression_level=None, row_group_size=1000):
    from .data.fileio import _read_root, _read_hdf5, _read_awkd

    ext = os.path.splitext(input_path)[1]
    if ext == ".root":
        table = _read_root(input_path, branches, treename=treename, branch_magic=branch_magic)
    elif ext == ".h5":
        table = _read_hdf5(input_path, branches)
    elif ext == ".awkd":
        table = _read_awkd(input_path, branches)
    else:
        raise RuntimeError(f"Unsupported format: {ext}")

    ak.to_parquet(
        table,
        output_path,
        compression=compression if compression != "none" else None,
        compression_level=compression_level,
        row_group_size=row_group_size,
    )
    return len(table)


def main():
    parser = argparse.ArgumentParser(description="Convert ROOT/HDF5/awkd files to Parquet")
    parser.add_argument("inputs", nargs="+",
                        help="Input files or glob patterns. If a .txt file is given, read paths from it.")
    parser.add_argument("-o", "--output-dir", required=True,
                        help="Output directory for parquet files")
    parser.add_argument("--data-config", default=None,
                        help="Data config YAML to determine which branches to convert (default: all branches)")
    parser.add_argument("--compression", default="lz4", choices=["lz4", "zstd", "snappy", "gzip", "none"],
                        help="Compression codec (default: lz4)")
    parser.add_argument("--compression-level", type=int, default=None,
                        help="Compression level (default: codec default)")
    parser.add_argument("--row-group-size", type=int, default=1000,
                        help="Row group size in events (default: 1000)")
    parser.add_argument("--treename", default=None,
                        help="ROOT tree name (auto-detected if not specified)")
    parser.add_argument("-j", "--workers", type=int, default=1,
                        help="Number of parallel workers (default: 1)")
    args = parser.parse_args()

    # Resolve input files
    input_files = []
    for inp in args.inputs:
        if inp.endswith(".txt"):
            with open(inp) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        input_files.extend(glob.glob(line) if "*" in line else [line])
        else:
            expanded = glob.glob(inp) if "*" in inp else [inp]
            input_files.extend(expanded)
    input_files = sorted(set(input_files))

    if not input_files:
        parser.error("No input files found")

    # Determine branches
    branches = None
    branch_magic = None
    if args.data_config:
        from .data.config import DataConfig
        dc = DataConfig.load(args.data_config)
        branches = list(dc.test_load_branches)
        branch_magic = dc.branch_magic
        print(f"Using {len(branches)} branches from {args.data_config}")
    else:
        import uproot
        with uproot.open(input_files[0]) as f:
            treename = args.treename
            if treename is None:
                treename = [k.split(";")[0] for k, v in f.items() if getattr(v, "classname", "") == "TTree"][0]
            branches = list(f[treename].keys())
        print(f"Converting all {len(branches)} branches")

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Input: {len(input_files)} files")
    print(f"Output: {args.output_dir}")
    print(f"Compression: {args.compression}, row_group_size: {args.row_group_size}")

    def _convert_one(input_path):
        basename = os.path.splitext(os.path.basename(input_path))[0] + ".parquet"
        output_path = os.path.join(args.output_dir, basename)
        n = convert_file(
            input_path, output_path,
            branches=branches,
            treename=args.treename,
            branch_magic=branch_magic,
            compression=args.compression,
            compression_level=args.compression_level,
            row_group_size=args.row_group_size,
        )
        in_sz = os.path.getsize(input_path) / 1e6
        out_sz = os.path.getsize(output_path) / 1e6
        return n, in_sz, out_sz

    t0 = time.perf_counter()
    total_events = 0
    total_in = 0
    total_out = 0

    if args.workers > 1:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_convert_one, f): f for f in input_files}
            with tqdm.tqdm(total=len(input_files)) as pbar:
                for future in as_completed(futures):
                    n, in_sz, out_sz = future.result()
                    total_events += n
                    total_in += in_sz
                    total_out += out_sz
                    pbar.update(1)
    else:
        for input_path in tqdm.tqdm(input_files):
            n, in_sz, out_sz = _convert_one(input_path)
            total_events += n
            total_in += in_sz
            total_out += out_sz

    elapsed = time.perf_counter() - t0
    print(f"\nConverted {len(input_files)} files, {total_events:,} events in {elapsed:.1f}s")
    print(f"Size: {total_in:.0f} MB -> {total_out:.0f} MB ({total_out/total_in*100:.0f}%)")


if __name__ == "__main__":
    main()
