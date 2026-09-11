"""How ``ParticleTransformer`` combines the pair bias with the key-padding mask.

``_forward_encoder`` adds the two together once and hands the same tensor to every block,
rather than letting each block's ``Attention.forward`` redo the sum. These tests pin the
invariants that make that legal: pre-merging equals merging inside ``Attention``; blocks
that scale the bias by ``c_mask`` still take the per-block path; and padded positions
never influence the real ones.
"""

import unittest

import torch
from torch.utils._python_dispatch import TorchDispatchMode

from weaver.nn.model.ParticleTransformer import Attention, ParticleTransformer

_INPUT_DIM = 12
_NUM_CLASSES = 4
_EMBED_DIM = 32
_NUM_HEADS = 4

_PART_KWARGS = dict(
    input_dim=_INPUT_DIM,
    num_classes=_NUM_CLASSES,
    embed_dims=[_EMBED_DIM],
    pair_embed_dims=(8, 8),
    num_heads=_NUM_HEADS,
    num_layers=3,
    num_cls_layers=1,
    block_params={"ffn_ratio": 2, "dropout": 0, "attn_dropout": 0, "activation_dropout": 0},
    fix_init=False,
    trim=False,  # keep `SequenceTrimmer`'s warmup counter out of the comparisons
)


def _four_vectors(batch, seq_len, g):
    """(px, py, pz, E) with E = sqrt(p^2 + m^2).

    The pairwise features take ``log`` of quantities like ``1 + 2 pz / (E - pz)``, which
    is only defined for on-shell vectors -- plain ``randn`` 4-vectors give NaN.
    """
    p3 = torch.randn(batch, 3, seq_len, generator=g)
    m = torch.rand(batch, 1, seq_len, generator=g) * 0.2
    energy = (p3.square().sum(dim=1, keepdim=True) + m.square()).sqrt()
    return torch.cat((p3, energy), dim=1)


def _inputs(batch=3, seq_len=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(batch, _INPUT_DIM, seq_len, generator=g)
    v = _four_vectors(batch, seq_len, g)
    mask = torch.zeros(batch, 1, seq_len)
    for i in range(batch):
        mask[i, 0, : min(seq_len, 4 + 5 * i)] = 1
    return x, v, mask


def _make_model(seed=0, **kwargs):
    torch.manual_seed(seed)
    cfg = dict(_PART_KWARGS)
    cfg.update(kwargs)
    return ParticleTransformer(**cfg).eval()


class _CountDenseMasks(TorchDispatchMode):
    """Counts 4-D additions, i.e. every ``(batch, num_heads, seq_len, seq_len)`` mask the
    forward pass materializes -- plus a constant offset from the rest of the model, which
    the tests below cancel out by comparing configurations."""

    def __init__(self):
        self.n = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        if func is torch.ops.aten.add.Tensor and isinstance(out, torch.Tensor) and out.dim() == 4:
            self.n += 1
        return out


def _count_dense_masks(model, x, v, mask):
    with torch.no_grad(), _CountDenseMasks() as counter:
        model(x, v=v, mask=mask)
    return counter.n


class PreMergedAttnMaskTest(unittest.TestCase):
    def assert_padding_invariant(self, model, x, v, mask, rtol=1e-4, atol=1e-5):
        """Each event padded out to the batch length must give the same answer as that
        event on its own, with no padding."""
        lengths = mask.squeeze(1).sum(dim=-1).long().tolist()
        with torch.no_grad():
            padded = model(x, v=v, mask=mask)
            for i, n in enumerate(lengths):
                trimmed = model(x[i : i + 1, :, :n], v=v[i : i + 1, :, :n], mask=mask[i : i + 1, :, :n])
                torch.testing.assert_close(padded[i : i + 1], trimmed, rtol=rtol, atol=atol)

    def test_attention_premerged_mask_matches_key_padding_mask(self):
        """The invariant the shared mask rests on: adding the padding ``-inf`` to the bias
        up front is the same as passing the padding mask separately."""
        torch.manual_seed(0)
        batch, seq_len = 2, 12
        attn = Attention(_EMBED_DIM, _NUM_HEADS).eval()
        q = torch.randn(batch, seq_len, _EMBED_DIM)
        bias = torch.randn(batch, _NUM_HEADS, seq_len, seq_len)
        padding_mask = torch.zeros(batch, seq_len, dtype=torch.bool)
        padding_mask[0, 7:] = True
        padding_mask[1, 5:] = True

        per_block = attn(q, q, q, key_padding_mask=padding_mask, attn_mask=bias)[0]

        pad_bias = torch.zeros_like(padding_mask, dtype=bias.dtype)
        pad_bias = pad_bias.masked_fill(padding_mask, float("-inf"))
        merged = bias + pad_bias.view(batch, 1, 1, seq_len)
        shared = attn(q, q, q, key_padding_mask=None, attn_mask=merged)[0]

        torch.testing.assert_close(shared, per_block)

    def test_c_mask_blocks_take_the_unmerged_path(self):
        """The two paths must compute the same function.

        ``c_mask`` is 1 at initialization, so a ``scale_attn_mask=True`` model (per-block
        path) and a default one (shared mask) differ only in how they get there.
        """
        x, v, mask = _inputs()
        shared = _make_model(block_params=dict(_PART_KWARGS["block_params"], scale_attn_mask=False))
        unmerged = _make_model(block_params=dict(_PART_KWARGS["block_params"], scale_attn_mask=True))
        self.assertIsNone(shared.blocks[0].c_mask)
        self.assertIsNotNone(unmerged.blocks[0].c_mask)
        unmerged.load_state_dict(shared.state_dict(), strict=False)
        for block in unmerged.blocks:
            self.assertTrue(torch.equal(block.c_mask, torch.ones(1)))

        with torch.no_grad():
            torch.testing.assert_close(unmerged(x, v=v, mask=mask), shared(x, v=v, mask=mask))

    def test_padding_does_not_change_the_result(self):
        """The masking invariant itself.

        Unlike perturbing the padded slots (below), this varies *how many* padded keys the
        attention sees, so it actually exercises the padding ``-inf``.
        """
        x, v, mask = _inputs()
        for kwargs in ({}, {"include_global_token": True, "num_cls_layers": 0}, {"num_cls_layers": 0}):
            with self.subTest(**kwargs):
                self.assert_padding_invariant(_make_model(**kwargs), x, v, mask)

    def test_negative_c_mask_still_masks_the_padding(self):
        """Why ``c_mask`` blocks keep the per-block path.

        Applied to a pre-merged mask, a negative ``c_mask`` would flip the padding
        ``-inf`` to ``+inf`` and point the attention straight at the padded tokens.
        """
        x, v, mask = _inputs()
        model = _make_model(block_params=dict(_PART_KWARGS["block_params"], scale_attn_mask=True))
        with torch.no_grad():
            for block in model.blocks:
                block.c_mask.fill_(-0.7)
        self.assert_padding_invariant(model, x, v, mask)

    def test_output_is_independent_of_padded_inputs(self):
        """End-to-end masking check: garbage in the padded slots cannot leak into the
        output, whichever path the blocks take."""
        x, v, mask = _inputs()
        g = torch.Generator().manual_seed(7)
        noise_x = torch.randn(x.shape, generator=g) * 50
        # the junk in the padded slots still has to be on-shell: the pairwise features
        # are computed densely for every pair, and a NaN there would survive the mask
        # (0 * NaN is NaN) and so would flag masking bugs that aren't there
        noise_v = _four_vectors(v.size(0), v.size(-1), g) * 50
        keep = mask.bool()
        x_perturbed = torch.where(keep, x, noise_x)
        v_perturbed = torch.where(keep, v, noise_v)

        for kwargs in ({}, {"include_global_token": True, "num_cls_layers": 0}, {"num_cls_layers": 0}):
            with self.subTest(**kwargs):
                model = _make_model(**kwargs)
                with torch.no_grad():
                    ref = model(x, v=v, mask=mask)
                    out = model(x_perturbed, v=v_perturbed, mask=mask)
                torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)

    def test_blocks_without_pair_bias_keep_the_padding_mask(self):
        """Blocks excluded by ``block_ids_with_attn_mask`` never carried a dense mask, so
        they still get the (batch, seq_len) padding mask -- and must stay masked."""
        x, v, mask = _inputs()
        model = _make_model(block_ids_with_attn_mask=[0])
        seen = []
        original = model.blocks[1].forward

        def spy(*args, **kwargs):
            seen.append((kwargs.get("padding_mask"), kwargs.get("attn_mask")))
            return original(*args, **kwargs)

        model.blocks[1].forward = spy
        with torch.no_grad():
            model(x, v=v, mask=mask)
        (padding_mask, attn_mask), = seen
        self.assertIsNone(attn_mask)
        self.assertIsNotNone(padding_mask)
        self.assertEqual(padding_mask.shape, (x.size(0), x.size(-1)))

    def test_partial_block_ids_still_share_the_merged_mask(self):
        """Only *some* blocks taking the pair bias does not stop the rest from sharing.

        An excluded block is handed no bias at all, so it is no reason to keep the
        un-merged one alive for the whole encoder.
        """
        x, v, mask = _inputs()
        model = _make_model(block_ids_with_attn_mask=[0, 2])
        self.assertEqual(model.block_ids_with_attn_mask, [True, False, True])
        self.assertTrue(model._blocks_share_attn_mask)

        seen = {}
        for idx, block in enumerate(model.blocks):
            def spy(*args, _idx=idx, _fwd=block.forward, **kwargs):
                seen[_idx] = (kwargs.get("padding_mask"), kwargs.get("attn_mask"))
                return _fwd(*args, **kwargs)

            block.forward = spy
        with torch.no_grad():
            model(x, v=v, mask=mask)

        for idx in (0, 2):
            padding_mask, attn_mask = seen[idx]
            self.assertIsNone(padding_mask, f"block {idx} should carry the padding in the merged mask")
            self.assertIsNotNone(attn_mask)
        # the same tensor, not one merge per block
        self.assertIs(seen[0][1], seen[2][1])
        padding_mask, attn_mask = seen[1]
        self.assertIsNone(attn_mask, "an excluded block must not be given the pair bias")
        self.assertTrue(torch.equal(padding_mask, ~mask.bool().squeeze(1)))

        self.assert_padding_invariant(_make_model(block_ids_with_attn_mask=[0, 2]), x, v, mask)

    def test_c_mask_on_a_bias_block_keeps_the_unmerged_bias_alive(self):
        """A block that scales the bias is the one case still needing the un-merged one."""
        model = _make_model(block_params=dict(_PART_KWARGS["block_params"], scale_attn_mask=True))
        self.assertFalse(model._blocks_share_attn_mask)

    def test_shared_mask_is_built_once_no_matter_how_deep(self):
        """The point of the whole thing: one dense mask per forward, not one per block."""
        x, v, mask = _inputs()
        counts = {nl: _count_dense_masks(_make_model(num_layers=nl), x, v, mask) for nl in (3, 6, 9)}
        self.assertEqual(len(set(counts.values())), 1, f"mask count grows with depth: {counts}")

    def test_no_shared_mask_is_built_when_no_block_can_use_it(self):
        """With a `c_mask` on every block nothing can read the shared mask, so building it
        would just pin a dense tensor for the whole encoder."""
        x, v, mask = _inputs()
        c_mask_params = dict(_PART_KWARGS["block_params"], scale_attn_mask=True)
        for num_layers in (3, 6):
            with self.subTest(num_layers=num_layers):
                shared = _count_dense_masks(_make_model(num_layers=num_layers), x, v, mask)
                unmerged = _count_dense_masks(
                    _make_model(num_layers=num_layers, block_params=c_mask_params), x, v, mask
                )
                # the shared model builds 1 mask; the `c_mask` one builds `num_layers` of
                # them (one per block) and must not build a shared one on top
                self.assertEqual(unmerged, shared - 1 + num_layers)

    def test_pair_bias_gradient_flows_through_the_shared_mask(self):
        """The shared mask is used by several blocks, so the pair embedding's gradient
        must accumulate over all of them rather than only the last."""
        x, v, mask = _inputs()
        model = _make_model().train()
        model(x, v=v, mask=mask).square().mean().backward()
        grads = [p.grad for p in model.pair_embed.parameters() if p.grad is not None]
        self.assertTrue(grads)
        self.assertTrue(any(g.abs().sum() > 0 for g in grads))


if __name__ == "__main__":
    unittest.main()
