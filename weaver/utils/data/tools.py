import numpy as np
import awkward as ak
import numba


def _hash(*args):
    return np.array([x.__hash__() for x in zip(*args)])


def _concat(arrays, axis=0):
    if len(arrays) == 0:
        return np.array([])
    if isinstance(arrays[0], np.ndarray):
        return np.concatenate(arrays, axis=axis)
    else:
        return ak.concatenate(arrays, axis=axis)


def _stack(arrays, axis=1):
    if len(arrays) == 0:
        return np.array([])
    if isinstance(arrays[0], np.ndarray):
        return np.stack(arrays, axis=axis)
    else:
        s = [slice(None)] * (arrays[0].ndim + 1)
        s[axis] = np.newaxis
        s = tuple(s)
        return ak.concatenate([a.__getitem__(s) for a in arrays], axis=axis)


@numba.njit(cache=True)
def _pad_jagged_kernel(content, offsets, out):
    """Fill pre-allocated output (nrows x maxlen) from jagged flat content + offsets."""
    maxlen = out.shape[1]
    for i in range(len(offsets) - 1):
        start = offsets[i]
        stop = offsets[i + 1]
        n = stop - start
        if n > maxlen:
            n = maxlen
        for j in range(n):
            out[i, j] = content[start + j]


@numba.njit(cache=True)
def _repeat_pad_jagged_kernel(content, offsets, out):
    """Fill output (nrows x maxlen) by repeating each row's values cyclically."""
    maxlen = out.shape[1]
    for i in range(len(offsets) - 1):
        start = offsets[i]
        length = offsets[i + 1] - start
        if length == 0:
            continue
        for j in range(maxlen):
            out[i, j] = content[start + j % length]


def _get_content_and_offsets(a):
    """Extract flat content and offsets from a 1-level jagged awkward array. Returns None if not applicable."""
    try:
        layout = a.layout
        # unwrap wrappers that don't have offsets (e.g. VirtualArray, IndexedArray)
        while not hasattr(layout, "offsets"):
            if hasattr(layout, "content"):
                layout = layout.content
            else:
                return None
        offsets = np.asarray(layout.offsets.data)
        inner = layout.content
        # get the raw numpy data from the NumpyArray content
        content = np.asarray(inner.data)
        if content.ndim != 1:
            return None
        return content, offsets
    except Exception:
        return None


def _pad(a, maxlen, value=0, dtype="float32"):
    if isinstance(a, np.ndarray) and a.ndim >= 2 and a.shape[1] == maxlen:
        return a
    elif isinstance(a, ak.Array):
        if a.ndim == 1:
            a = ak.unflatten(a, 1)
        result = _get_content_and_offsets(a)
        if result is not None:
            content, offsets = result
            nrows = len(offsets) - 1
            out = np.full((nrows, maxlen), value, dtype=dtype)
            _pad_jagged_kernel(content.astype(dtype), offsets, out)
            return out
        # fallback for complex layouts
        a = ak.fill_none(ak.pad_none(a, maxlen, clip=True), value)
        return ak.values_astype(a, dtype)
    else:
        x = np.full((len(a), maxlen), value, dtype=dtype)
        for idx, s in enumerate(a):
            n = min(len(s), maxlen)
            if n > 0:
                x[idx, :n] = np.asarray(s[:n], dtype=dtype)
        return x


def _repeat_pad(a, maxlen, dtype="float32"):
    assert isinstance(a, ak.Array)
    if a.ndim == 1:
        a = ak.unflatten(a, 1)
    content, offsets = _get_content_and_offsets(a)
    nrows = len(offsets) - 1
    out = np.empty((nrows, maxlen), dtype=dtype)
    _repeat_pad_jagged_kernel(content.astype(dtype), offsets, out)
    return out


def _clip(a, a_min, a_max):
    if isinstance(a, np.ndarray) or a.ndim == 1:
        return np.clip(a, a_min, a_max)
    else:
        return ak.unflatten(np.clip(ak.to_numpy(ak.flatten(a)), a_min, a_max), ak.num(a))


def _knn(support, query, k, n_jobs=1):
    from scipy.spatial import cKDTree

    kdtree = cKDTree(support)
    d, idx = kdtree.query(query, k, n_jobs=n_jobs)
    return idx


def _batch_knn(supports, queries, k, maxlen_s, maxlen_q=None, n_jobs=1):
    assert len(supports) == len(queries)
    if maxlen_q is None:
        maxlen_q = maxlen_s
    batch_knn_idx = np.ones((len(supports), maxlen_q, k), dtype="int32") * (maxlen_s - 1)
    for i, (s, q) in enumerate(zip(supports, queries)):
        batch_knn_idx[i, : len(q[:maxlen_q]), :] = _knn(s[:maxlen_s], q[:maxlen_q], k, n_jobs=n_jobs).reshape(
            (-1, k)
        )  # (len(q), k)
    return batch_knn_idx


def _batch_permute_indices(array):
    random_array = ak.unflatten(np.random.rand(ak.count(array)), ak.num(array))
    return ak.argsort(random_array)


def _batch_argsort(array):
    return ak.argsort(array)


def _batch_gather(array, indices):
    return array[indices]


def _batch_permute_and_drop_indices(array, random_permute=True, drop_rate_min=0, drop_rate_max=0, min_elements=1):
    indices = _batch_permute_indices(array) if random_permute else _batch_argsort(array)
    if drop_rate_min:
        counts = ak.num(array)
        assert 0 < drop_rate_min <= drop_rate_max < 1
        drop_rate = np.random.uniform(low=drop_rate_min, high=drop_rate_max, size=len(counts))
        counts = np.maximum(min_elements, np.round(counts * (1 - drop_rate)))
        counts = ak.values_astype(counts, "int64")
        indices = indices[indices < counts]
    return indices


@numba.njit(cache=True)
def _fused_repeat_pad_one_var(content, offsets, center, scale, lo, hi, do_center, out_v):
    """Standardize + clip + repeat-pad + nan_to_num for one variable, writing into a 2D slice (nrows, maxlen)."""
    nrows = len(offsets) - 1
    maxlen = out_v.shape[1]
    for i in range(nrows):
        row_start = offsets[i]
        row_len = offsets[i + 1] - row_start
        if row_len == 0:
            continue
        for j in range(maxlen):
            val = np.float32(content[row_start + j % row_len])
            if do_center:
                val = (val - center) * scale
                if val < lo:
                    val = lo
                elif val > hi:
                    val = hi
            if val != val:
                val = np.float32(0.0)
            out_v[i, j] = val


@numba.njit(cache=True)
def _fused_constant_pad_one_var(content, offsets, center, scale, lo, hi, do_center, pad_value, out_v):
    """Standardize + clip + constant-pad + nan_to_num for one variable, writing into a 2D slice (nrows, maxlen)."""
    nrows = len(offsets) - 1
    maxlen = out_v.shape[1]
    for i in range(nrows):
        row_start = offsets[i]
        row_len = offsets[i + 1] - row_start
        n = min(row_len, maxlen)
        for j in range(n, maxlen):
            out_v[i, j] = pad_value
        for j in range(n):
            val = np.float32(content[row_start + j])
            if do_center:
                val = (val - center) * scale
                if val < lo:
                    val = lo
                elif val > hi:
                    val = hi
            if val != val:
                val = np.float32(0.0)
            out_v[i, j] = val


def _fused_pad_and_stack(table, var_names, preprocess_params, dtype="float32"):
    """Fused standardize + clip + pad + nan_to_num + stack for variables sharing the same jagged structure.

    Processes each variable with a per-variable numba kernel but writes directly into slices of
    the final (nrows, n_vars, padlen) output array, avoiding intermediate allocations and the
    separate stacking + nan_to_num passes.
    """
    if not var_names:
        return None

    first_result = _get_content_and_offsets(table[var_names[0]])
    if first_result is None:
        return None
    _, shared_offsets = first_result

    nrows = len(shared_offsets) - 1
    n_vars = len(var_names)
    padlen = preprocess_params[var_names[0]]["length"]
    if padlen is None:
        return None

    pad_mode = preprocess_params[var_names[0]]["pad_mode"]
    out = np.zeros((nrows, n_vars, padlen), dtype=dtype) if pad_mode != "wrap" else np.empty((nrows, n_vars, padlen), dtype=dtype)

    for vi, vn in enumerate(var_names):
        result = _get_content_and_offsets(table[vn])
        if result is None or len(result[1]) != len(shared_offsets):
            return None
        content, offsets = result
        if offsets[-1] != shared_offsets[-1]:
            return None

        p = preprocess_params[vn]
        do_center = p["center"] is not None
        center = np.float32(p["center"]) if do_center else np.float32(0.0)
        scale = np.float32(p["scale"])
        lo = np.float32(p["min"])
        hi = np.float32(p["max"])
        content_f32 = content.astype(np.float32) if content.dtype != np.float32 else content

        if pad_mode == "wrap":
            _fused_repeat_pad_one_var(content_f32, offsets, center, scale, lo, hi, do_center, out[:, vi, :])
        else:
            _fused_constant_pad_one_var(content_f32, offsets, center, scale, lo, hi, do_center,
                                        np.float32(p["pad_value"]), out[:, vi, :])
    return out


def _p4_from_pxpypze(px, py, pz, energy):
    import vector

    vector.register_awkward()
    return vector.zip({"px": px, "py": py, "pz": pz, "energy": energy})


def _p4_from_ptetaphie(pt, eta, phi, energy):
    import vector

    vector.register_awkward()
    return vector.zip({"pt": pt, "eta": eta, "phi": phi, "energy": energy})


def _p4_from_ptetaphim(pt, eta, phi, mass):
    import vector

    vector.register_awkward()
    return vector.zip({"pt": pt, "eta": eta, "phi": phi, "mass": mass})
