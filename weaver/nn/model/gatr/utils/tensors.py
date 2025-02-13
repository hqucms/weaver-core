import torch


def assert_equal(vals):
    """Assert all values in sequence are equal."""
    for v in vals:
        assert v == vals[0]


def block_stack(tensors, dim1, dim2):
    """Block diagonally stack tensors along dimensions dim1 and dim2."""
    assert_equal([t.dim() for t in tensors])
    shapes = [t.shape for t in tensors]
    shapes_t = list(map(list, zip(*shapes)))
    for i, ss in enumerate(shapes_t):
        if i not in (dim1, dim2):
            assert_equal(ss)

    dim2_len = sum(shapes_t[dim2])
    opts = dict(device=tensors[0].device, dtype=tensors[0].dtype)

    padded_tensors = []
    offset = 0
    for tensor in tensors:
        before_shape = list(tensor.shape)
        before_shape[dim2] = offset
        after_shape = list(tensor.shape)
        after_shape[dim2] = dim2_len - tensor.shape[dim2] - offset
        before = torch.zeros(*before_shape, **opts)
        after = torch.zeros(*after_shape, **opts)
        padded = torch.cat([before, tensor, after], dim2)
        padded_tensors.append(padded)
        offset += tensor.shape[dim2]
    return torch.cat(padded_tensors, dim1)
