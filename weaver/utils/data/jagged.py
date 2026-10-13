"""Jagged (variable-length) storage and collation for the per-particle input groups.

The stock loader pads every event to the data config's ``length`` as soon as a chunk of
events is fetched, so a worker holds ``(events, vars, length)`` float32 however empty the
rows are. Here a chunk is kept as ``values (tokens, vars)`` + ``offsets (events + 1,)``
(:class:`JaggedArray`), and the mode decides where the padding happens:

``host``    pad to the config ``length`` when the batch is built in the worker: batches are
            bit-identical to the stock loader.
``device``  hand the batch over unpadded and pad it to the config ``length`` on the training
            device, after the host-to-device copy (:meth:`JaggedCollator.pad_on_device`): the
            model gets the same bit-identical batch, while every host-side copy (worker to main
            process, pinning, host to device) carries only the real particles.
``nested``  no padding at all: the model gets ``torch.nested`` jagged-layout tensors of shape
            ``(B, j, vars)`` and must be written for them.
"""

from typing import NamedTuple

import numba
import numpy as np
import torch
from torch.utils.data._utils.collate import default_collate

JAGGED_MODES = ("host", "device", "nested")


class Unpadded(NamedTuple):
    """A ``--jagged device`` input group in transit: ``values`` (tokens, vars) and ``offsets``
    (B + 1,). Plain tensors on purpose: the DataLoader pins a named tuple element-wise and
    ``.to()`` moves both, without the per-op Python overhead of a ``torch.nested`` tensor
    in the training loop's thread."""

    values: torch.Tensor
    offsets: torch.Tensor

    def to(self, *args, **kwargs):
        return Unpadded(self.values.to(*args, **kwargs), self.offsets.to(*args, **kwargs))


class JaggedRow(np.ndarray):
    """One event's ``(n_tokens, n_vars)`` slice of a :class:`JaggedArray`; the type is
    what tells the collator that the sample is still unpadded."""


class JaggedArray:
    """``values``: (tokens, n_vars) float32, token-major; ``offsets``: (events + 1,) int64."""

    __slots__ = ("values", "offsets")

    def __init__(self, values, offsets):
        self.values = values
        self.offsets = offsets

    def __len__(self):
        return len(self.offsets) - 1

    def __getitem__(self, i):
        return self.values[self.offsets[i] : self.offsets[i + 1]].view(JaggedRow)

    @property
    def nbytes(self):
        return self.values.nbytes + self.offsets.nbytes


@numba.njit(cache=True)
def _fused_jagged_kernel(all_content, offsets, new_offsets, centers, scales, los, his, do_centers, out):
    """Standardize + clip + nan_to_num + stack, keeping each row's first ``maxlen`` tokens.

    ``all_content`` is (n_vars, content_len); ``out`` is (tokens_kept, n_vars).
    """
    n_vars = all_content.shape[0]
    for i in range(len(offsets) - 1):
        src = offsets[i]
        dst = new_offsets[i]
        n = new_offsets[i + 1] - dst
        for vi in range(n_vars):
            center = centers[vi]
            scale_v = scales[vi]
            lo_v = los[vi]
            hi_v = his[vi]
            dc = do_centers[vi]
            for j in range(n):
                val = all_content[vi, src + j]
                if dc:
                    val = (val - center) * scale_v
                    if val < lo_v:
                        val = lo_v
                    elif val > hi_v:
                        val = hi_v
                if val != val:
                    val = np.float32(0.0)
                out[dst + j, vi] = val


def _fused_jagged_stack(table, var_names, preprocess_params):
    """Jagged counterpart of ``tools._fused_pad_and_stack``: same per-value arithmetic, no
    padding. Returns ``None`` when the variables do not share one plain jagged structure."""
    from .tools import _get_content_and_offsets

    if not var_names:
        return None
    padlen = preprocess_params[var_names[0]]["length"]
    if padlen is None:
        return None
    first = _get_content_and_offsets(table[var_names[0]])
    if first is None:
        return None
    shared_offsets = first[1]

    n_vars = len(var_names)
    content_len = int(shared_offsets[-1])
    all_content = np.empty((n_vars, content_len), dtype=np.float32)
    centers = np.zeros(n_vars, dtype=np.float32)
    scales = np.zeros(n_vars, dtype=np.float32)
    los = np.zeros(n_vars, dtype=np.float32)
    his = np.zeros(n_vars, dtype=np.float32)
    do_centers = np.zeros(n_vars, dtype=np.bool_)
    for vi, vn in enumerate(var_names):
        result = _get_content_and_offsets(table[vn])
        if result is None or len(result[1]) != len(shared_offsets):
            return None
        content, offsets = result
        if not np.array_equal(offsets, shared_offsets):
            return None
        p = preprocess_params[vn]
        dc = p["center"] is not None
        do_centers[vi] = dc
        centers[vi] = np.float32(p["center"]) if dc else np.float32(0.0)
        scales[vi] = np.float32(p["scale"])
        los[vi] = np.float32(p["min"])
        his[vi] = np.float32(p["max"])
        all_content[vi] = content

    offsets64 = shared_offsets.astype(np.int64, copy=False)
    counts = np.minimum(np.diff(offsets64), padlen)
    new_offsets = np.zeros(len(counts) + 1, dtype=np.int64)
    np.cumsum(counts, out=new_offsets[1:])
    out = np.empty((int(new_offsets[-1]), n_vars), dtype=np.float32)
    _fused_jagged_kernel(all_content, offsets64, new_offsets, centers, scales, los, his, do_centers, out)
    return JaggedArray(out, new_offsets)


@numba.njit(cache=True)
def _pad_batch_kernel(values, offsets, wrap, pad_values, out):
    """(tokens, n_vars) + offsets -> ``out`` (B, n_vars, L), wrap- or constant-padded."""
    n_vars = out.shape[1]
    maxlen = out.shape[2]
    for i in range(out.shape[0]):
        start = offsets[i]
        n = min(offsets[i + 1] - start, maxlen)
        for j in range(n):
            for vi in range(n_vars):
                out[i, vi, j] = values[start + j, vi]
        if wrap and n > 0:
            for j in range(n, maxlen):
                src = start + j % n
                for vi in range(n_vars):
                    out[i, vi, j] = values[src, vi]
        else:
            # an empty row is all zeros in the stock wrap kernel as well
            for vi in range(n_vars):
                pv = np.float32(0.0) if wrap else pad_values[vi]
                for j in range(n, maxlen):
                    out[i, vi, j] = pv


@numba.njit(cache=True)
def _gather_rows_kernel(values, offsets, idx, out_offsets, out):
    """Copy rows ``idx`` of a jagged (values, offsets) into ``out`` at ``out_offsets``."""
    n_vars = values.shape[1]
    for b in range(len(idx)):
        src = offsets[idx[b]]
        dst = out_offsets[b]
        for j in range(out_offsets[b + 1] - dst):
            for vi in range(n_vars):
                out[dst + j, vi] = values[src + j, vi]


def gather_rows(parts):
    """Rows of one or more :class:`JaggedArray` s as one batch.

    ``parts``: list of ``(JaggedArray, row indices)``, concatenated in order (a batch that
    spans two fetched chunks has two parts). Returns ``(values, offsets)``.
    """
    lengths = np.concatenate([a.offsets[idx + 1] - a.offsets[idx] for a, idx in parts])
    offsets = np.zeros(len(lengths) + 1, dtype=np.int64)
    np.cumsum(lengths, out=offsets[1:])
    values = np.empty((int(offsets[-1]), parts[0][0].values.shape[1]), dtype=np.float32)
    start = 0
    for a, idx in parts:
        _gather_rows_kernel(a.values, a.offsets, idx, offsets[start:], values)
        start += len(idx)
    return values, offsets


class JaggedCollator:
    """``collate_fn`` for samples drawn from a jagged-mode ``SimpleIterDataset``."""

    def __init__(self, data_config, mode):
        if mode not in JAGGED_MODES:
            raise ValueError(f"jagged mode must be one of {JAGGED_MODES}, got {mode!r}")
        self.mode = mode
        self.pad = {}
        for k, names in data_config.input_dicts.items():
            p = data_config.preprocess_params[names[0]]
            if p["length"] is None:
                continue
            pad_values = np.array(
                [data_config.preprocess_params[n].get("pad_value", 0) for n in names], dtype=np.float32
            )
            self.pad[k] = (p["length"], p["pad_mode"] == "wrap", pad_values)
        self._device_pad_values = {}

    def _collate_group(self, key, rows):
        lengths = np.fromiter((r.shape[0] for r in rows), dtype=np.int64, count=len(rows))
        offsets = np.zeros(len(rows) + 1, dtype=np.int64)
        np.cumsum(lengths, out=offsets[1:])
        return self.build(key, np.concatenate(rows).view(np.ndarray), offsets)

    def build(self, key, values, offsets):
        """The batch tensor of input group ``key`` from its (tokens, vars) values + offsets."""
        lengths = np.diff(offsets)
        if self.mode == "device":
            return Unpadded(torch.from_numpy(values), torch.from_numpy(offsets))
        if self.mode == "nested":
            return torch.nested.nested_tensor_from_jagged(
                torch.from_numpy(values),
                offsets=torch.from_numpy(offsets),
                min_seqlen=int(lengths.min()),
                max_seqlen=int(lengths.max()),
            )
        length, wrap, pad_values = self.pad[key]
        out = np.empty((len(lengths), values.shape[1], length), dtype=np.float32)
        _pad_batch_kernel(values, offsets, wrap, pad_values, out)
        return torch.from_numpy(out)

    def pad_on_device(self, key, x):
        """Pad the :class:`Unpadded` input group ``key`` to the stock ``(B, vars, length)``
        batch on whatever device it lives on.

        One gather, same rule as the host kernel: ``wrap`` repeats each event's particles
        cyclically (an empty event stays all zeros), ``constant`` writes the per-variable
        pad values. The values are copied, not recomputed, so the result is bit-identical to
        ``host`` mode. No device-to-host sync.
        """
        length, wrap, pad_values = self.pad[key]
        values, offsets = x
        n_vars = values.shape[1]
        starts = offsets[:-1].unsqueeze(1)
        lengths = (offsets[1:] - offsets[:-1]).unsqueeze(1)  # (B, 1), already <= length
        j = torch.arange(length, device=values.device).unsqueeze(0)  # (1, L)
        if wrap:
            idx = starts + j % lengths.clamp(min=1)
            valid = (lengths > 0).expand(-1, length)
        else:
            idx = starts + torch.minimum(j, (lengths - 1).clamp(min=0))
            valid = j < lengths
        if values.shape[0] == 0:
            # every event of the batch is empty: nothing to gather from
            values = values.new_zeros(1, n_vars)
        # an empty event points one past its start, which for a trailing empty event is past
        # the end; those slots are masked below, so any in-range row will do
        idx = idx.clamp(max=values.shape[0] - 1)
        out = values[idx]  # (B, L, vars)
        if wrap:
            fill = out.new_zeros(())
        else:
            fill = self._device_pad_values.get((key, values.device))
            if fill is None:
                fill = torch.as_tensor(pad_values, device=values.device)
                self._device_pad_values[(key, values.device)] = fill
        out = torch.where(valid.unsqueeze(-1), out, fill)
        return out.transpose(1, 2).contiguous()

    def __call__(self, batch):
        X = {}
        for k in batch[0][0]:
            rows = [sample[0][k] for sample in batch]
            if isinstance(rows[0], JaggedRow):
                X[k] = self._collate_group(k, rows)
            else:
                X[k] = default_collate(rows)
        y = default_collate([sample[1] for sample in batch])
        Z = default_collate([sample[2] for sample in batch])
        return X, y, Z
