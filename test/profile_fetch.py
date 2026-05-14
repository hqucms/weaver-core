#!/usr/bin/env python3
"""Profile a single _load_next + _finalize_inputs cycle to find hotspots."""

import glob
import time
import logging
import numpy as np
import awkward as ak

logging.basicConfig(level=logging.WARNING)

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from weaver.utils.data.config import DataConfig
from weaver.utils.data.fileio import _read_files
from weaver.utils.data.preprocess import _apply_selection, _build_new_variables, _build_weights
from weaver.utils.data.tools import (
    _pad, _repeat_pad, _clip,
    _fused_pad_and_stack, _get_content_and_offsets,
)
from weaver.utils.dataset import _finalize_inputs, _load_next, _get_reweight_indices
from functools import partial

data_config_file = "test/data/JetClass_full.yaml"
data_dir = "/data/hqu/datasets/JetClass/Pythia/val_5M"
files = sorted(glob.glob(f"{data_dir}/*.root"))
print(f"Files: {len(files)}")

load_range = (0.0, 0.05)

# Load separate configs for train (no observers) and test (with observers, no reweight)
train_dc = DataConfig.load(data_config_file, load_observers=False)
test_dc = DataConfig.load(data_config_file, load_reweight_info=False)

train_options = {
    "training": True, "mode": "train", "shuffle": True,
    "reweight": train_dc.weight_name is not None,
    "up_sample": True, "weight_scale": 1, "max_resample": 10, "in_memory": False,
}
test_options = {
    "training": False, "mode": "test", "shuffle": False,
    "reweight": False, "up_sample": False, "weight_scale": 1,
    "max_resample": 10, "in_memory": False,
}

# ============= WARM UP numba =============
print("Warming up numba JIT...")
t_w0 = time.perf_counter()
_ = _load_next(test_dc, files[:1], (0.0, 0.01), test_options)
t_w1 = time.perf_counter()
print(f"Warmup done in {t_w1 - t_w0:.1f}s.\n")

for mode_name, data_config, options in [
    ("TRAIN", train_dc, train_options),
    ("TEST", test_dc, test_options),
]:
    load_branches = data_config.train_load_branches if options["training"] else data_config.test_load_branches

    print("=" * 70)
    print(f"MODE: {mode_name}  (z_variables={data_config.z_variables})")
    print("=" * 70)

    # End-to-end
    te0 = time.perf_counter()
    table_e2e, indices_e2e = _load_next(data_config, files, load_range, options)
    te1 = time.perf_counter()
    print(f"  End-to-end _load_next: {te1 - te0:.3f}s  |  Output entries: {len(indices_e2e):,}")

    # ---- STAGE 1: _read_files ----
    t0 = time.perf_counter()
    table = _read_files(
        files, load_branches, load_range,
        treename=data_config.treename,
        branch_magic=data_config.branch_magic,
        file_magic=data_config.file_magic,
    )
    t1 = time.perf_counter()
    n_events_raw = len(table)
    print(f"\n  STAGE 1 _read_files:  {t1 - t0:.3f}s  |  Events: {n_events_raw:,}  |  Fields: {len(table.fields)}")

    # ---- STAGE 2: _preprocess ----
    t2 = time.perf_counter()
    table["aux_training_"] = options["mode"] == "train"
    sel = data_config.selection if options["training"] else data_config.test_time_selection
    table = _apply_selection(table, sel, funcs=data_config.var_funcs)
    t2b = time.perf_counter()
    var_funcs = data_config.train_var_funcs if options["training"] else data_config.test_var_funcs
    table = _build_new_variables(table, var_funcs)
    t2c = time.perf_counter()
    if options["reweight"] and data_config.weight_name is not None:
        wgts = _build_weights(table, data_config)
        indices = _get_reweight_indices(wgts)
    else:
        indices = np.arange(len(table))
    t2d = time.perf_counter()
    print(f"  STAGE 2 _preprocess:  {t2d - t2:.3f}s")
    print(f"    selection:         {t2b - t2:.4f}s")
    print(f"    build_new_vars:    {t2c - t2b:.4f}s  ({len(var_funcs)} vars)")
    print(f"    reweighting:       {t2d - t2c:.4f}s")

    # ---- STAGE 3: _finalize_inputs (breakdown) ----
    # Use table from above (already preprocessed)
    output = {}

    ta = time.perf_counter()
    for k in data_config.z_variables:
        if k in data_config.observer_names and k in table.fields:
            arr = table[k]
            output[k] = ak.to_numpy(arr) if isinstance(arr, ak.Array) and arr.ndim == 1 else arr
    tb = time.perf_counter()
    for k in data_config.label_names:
        output[k] = ak.to_numpy(table[k])
    tc = time.perf_counter()

    fused_vars = set()
    for group_name, var_names in data_config.input_dicts.items():
        if data_config.preprocess_params[var_names[0]]["length"] is None:
            continue
        tg0 = time.perf_counter()
        result = _fused_pad_and_stack(table, var_names, data_config.preprocess_params)
        tg1 = time.perf_counter()
        if result is not None:
            output["_" + group_name] = result
            fused_vars.update(var_names)
            print(f"    fused {group_name:20s}: {tg1 - tg0:.4f}s  ({len(var_names):2d} vars, shape={result.shape})")
    td = time.perf_counter()

    n_fallback = 0
    for k, params in data_config.preprocess_params.items():
        if k in fused_vars:
            continue
        n_fallback += 1
        tf0 = time.perf_counter()
        if params["center"] is not None:
            table[k] = _clip((table[k] - params["center"]) * params["scale"], params["min"], params["max"])
        if params["length"] is not None:
            pad_fn = _repeat_pad if params["pad_mode"] == "wrap" else partial(_pad, value=params["pad_value"])
            table[k] = pad_fn(table[k], params["length"])
        table[k] = np.nan_to_num(table[k])
        tf1 = time.perf_counter()
        if tf1 - tf0 > 0.01:
            print(f"    fallback {k:25s}: {tf1 - tf0:.4f}s")
    te = time.perf_counter()

    def _to_f32(x):
        if isinstance(x, np.ndarray):
            return x if x.dtype == np.float32 else x.astype("float32")
        return np.asarray(ak.to_numpy(ak.values_astype(x, "float32")), dtype="float32")

    for k, names in data_config.input_dicts.items():
        if "_" + k in output:
            continue
        if len(names) == 1 and data_config.preprocess_params[names[0]]["length"] is None:
            output["_" + k] = _to_f32(table[names[0]])
        else:
            first = _to_f32(table[names[0]])
            result = np.empty((len(first), len(names)) + first.shape[1:], dtype="float32")
            result[:, 0] = first
            for idx, n in enumerate(names[1:], 1):
                result[:, idx] = _to_f32(table[n])
            output["_" + k] = result
    tff = time.perf_counter()

    for k in data_config.z_variables:
        if k in data_config.monitor_variables and k in table.fields:
            arr = table[k]
            output[k] = ak.to_numpy(arr) if isinstance(arr, ak.Array) and arr.ndim == 1 else arr
    tg = time.perf_counter()

    print(f"  STAGE 3 _finalize:    {tg - ta:.3f}s")
    print(f"    observers:       {tb - ta:.4f}s")
    print(f"    labels:          {tc - tb:.4f}s")
    print(f"    fused total:     {td - tc:.4f}s")
    print(f"    fallback vars:   {te - td:.4f}s  ({n_fallback} vars)")
    print(f"    stack remaining: {tff - te:.4f}s")
    print(f"    monitors:        {tg - tff:.4f}s")

    # ---- STAGE 4: get_data ----
    input_names = data_config.input_names
    label_names = data_config.label_names
    z_vars = [k for k in data_config.z_variables if k in output]
    t4 = time.perf_counter()
    for _ in range(1000):
        i = indices[np.random.randint(len(indices))]
        X = {k: output["_" + k][i] for k in input_names}
        y = {k: output[k][i] for k in label_names}
        Z = {k: output[k][i] for k in z_vars}
    t4f = time.perf_counter()
    print(f"  STAGE 4 get_data:     {(t4f - t4)/1000*1e6:.1f} μs/call (1000 calls)")

    # Summary
    t_read = t1 - t0
    t_prep = t2d - t2
    t_fin = tg - ta
    total = t_read + t_prep + t_fin
    print(f"\n  SUMMARY ({n_events_raw:,} events)")
    print(f"    _read_files:       {t_read:>7.3f}s  ({t_read/total*100:>5.1f}%)")
    print(f"    _preprocess:       {t_prep:>7.3f}s  ({t_prep/total*100:>5.1f}%)")
    print(f"    _finalize_inputs:  {t_fin:>7.3f}s  ({t_fin/total*100:>5.1f}%)")
    print(f"    TOTAL:             {total:>7.3f}s")

    print(f"\n  Output shapes:")
    for k, v in sorted(output.items()):
        if isinstance(v, np.ndarray):
            print(f"    {k}: {v.shape} {v.dtype}")
    print()

# === EXTRA: ak.concatenate + build_new_variables ===
print("=" * 70)
print("EXTRA: ak.concatenate cost")
load_branches = train_dc.train_load_branches
per_file = []
for f in files:
    t = _read_files([f], load_branches, load_range,
                    treename=train_dc.treename,
                    branch_magic=train_dc.branch_magic,
                    file_magic=train_dc.file_magic)
    per_file.append(t)
tc0 = time.perf_counter()
merged = ak.concatenate(per_file, axis=0)
tc1 = time.perf_counter()
print(f"  ak.concatenate({len(per_file)} arrays, {len(merged):,} events): {tc1 - tc0:.4f}s")

print("\nEXTRA: build_new_variables top-10 slowest:")
table = _read_files(files, load_branches, load_range,
                    treename=train_dc.treename,
                    branch_magic=train_dc.branch_magic,
                    file_magic=train_dc.file_magic)
table["aux_training_"] = True
from weaver.utils.data.eval_utils import _eval_expr
var_times = []
for k, expr in train_dc.train_var_funcs.items():
    if k in table.fields:
        continue
    tv0 = time.perf_counter()
    table[k] = _eval_expr(expr, table)
    tv1 = time.perf_counter()
    var_times.append((k, tv1 - tv0, expr))
var_times.sort(key=lambda x: -x[1])
for k, t, expr in var_times[:10]:
    print(f"    {k:30s} {t:.4f}s  expr={expr[:70]}")

print("\nDone.")
