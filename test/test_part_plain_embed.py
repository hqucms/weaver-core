"""Unit tests for ``ParticleTransformer``'s ``use_plain_embed`` input embedding.

``use_plain_embed=True`` reduces ``Embed`` to a bare ``nn.Linear``, matching the single
``linear_in`` of :class:`weaver.nn.model.PlainTransformer.PlainTransformerTagger`, so the
two models can be compared without the input-normalization confound.
"""

import unittest

import torch
import torch.nn as nn

from weaver.nn.model.ParticleTransformer import Embed, ParticleTransformer
from weaver.nn.model.PlainTransformer import PlainTransformerTagger

_INPUT_DIM = 24
_NUM_CLASSES = 4
_EMBED_DIM = 128

# the `-o` block that strips ParticleTransformer down to a plain transformer
_PART_KWARGS = dict(
    input_dim=_INPUT_DIM,
    num_classes=_NUM_CLASSES,
    embed_dims=[_EMBED_DIM],
    pair_embed_dims=None,
    num_heads=8,
    num_layers=8,
    num_cls_layers=0,
    block_params={"ffn_ratio": 2},
    version=3,
    weight_init=None,
    fix_init=False,
    include_global_token=True,
)


def _inputs(batch=3, seq_len=40, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(batch, _INPUT_DIM, seq_len, generator=g)
    v = torch.randn(batch, 4, seq_len, generator=g)
    mask = torch.zeros(batch, 1, seq_len)
    for i in range(batch):
        mask[i, 0, : min(seq_len, 5 + 7 * i)] = 1
    return x, v, mask


class PlainEmbedTest(unittest.TestCase):
    def test_plain_embed_is_a_single_linear(self):
        embed = Embed(_INPUT_DIM, [_EMBED_DIM], use_plain_embed=True)

        self.assertIsNone(embed.input_bn)
        self.assertEqual(len(embed.embed), 1)
        self.assertIsInstance(embed.embed[0], nn.Linear)
        # no BatchNorm / LayerNorm / activation anywhere
        for m in embed.modules():
            self.assertNotIsInstance(m, (nn.BatchNorm1d, nn.LayerNorm, nn.GELU, nn.ReLU))

    def test_default_embed_is_unchanged(self):
        embed = Embed(_INPUT_DIM, [_EMBED_DIM])

        self.assertIsInstance(embed.input_bn, nn.BatchNorm1d)
        self.assertIsInstance(embed.embed[0], nn.LayerNorm)
        self.assertIsInstance(embed.embed[1], nn.Linear)
        self.assertIsInstance(embed.embed[2], nn.GELU)

    def test_plain_embed_transposes_to_channels_last(self):
        # regression guard: without the input BatchNorm, `Embed.forward` used to skip
        # the (N, C, P) -> (N, P, C) transpose and apply the linear over the wrong axis
        embed = Embed(_INPUT_DIM, [_EMBED_DIM], use_plain_embed=True).eval()
        x = torch.randn(3, _INPUT_DIM, 40)

        out = embed(x)

        self.assertEqual(out.shape, (3, 40, _EMBED_DIM))
        torch.testing.assert_close(out, embed.embed[0](x.transpose(1, 2)))

    def test_matches_plain_transformer_linear_in(self):
        part = ParticleTransformer(**_PART_KWARGS, use_plain_embed=True)
        plain = PlainTransformerTagger(input_dim=_INPUT_DIM, num_classes=_NUM_CLASSES)

        part_linear = part.embed.embed[0]
        plain_linear = plain.net.linear_in

        self.assertEqual(part_linear.out_features, plain_linear.out_features)
        # PlainTransformer carries one extra input channel: the global-token flag,
        # which ParticleTransformer replaces with a learned `cls_token`
        self.assertEqual(part_linear.in_features + 1, plain_linear.in_features)
        self.assertIsNotNone(part_linear.bias)
        self.assertIsNotNone(plain_linear.bias)

    def test_forward_train_and_eval(self):
        x, v, mask = _inputs()
        for use_plain_embed in (False, True):
            model = ParticleTransformer(**_PART_KWARGS, use_plain_embed=use_plain_embed)
            for mode in ("train", "eval"):
                with self.subTest(use_plain_embed=use_plain_embed, mode=mode):
                    getattr(model, mode)()
                    with torch.no_grad():
                        out = model(x, v, mask)
                    self.assertEqual(out.shape, (x.size(0), _NUM_CLASSES))
                    self.assertTrue(torch.isfinite(out).all())

    def test_plain_embed_has_no_batchnorm_running_state(self):
        # no running stats => no train/eval discrepancy, as for PlainTransformer
        model = ParticleTransformer(**_PART_KWARGS, use_plain_embed=True)
        x, v, mask = _inputs()

        model.train()
        with torch.no_grad():
            model(x, v, mask)  # would update BatchNorm running stats
        model.eval()
        with torch.no_grad():
            train_then_eval = model(x, v, mask)
        with torch.no_grad():
            again = model(x, v, mask)

        torch.testing.assert_close(train_then_eval, again)
        self.assertEqual([k for k in model.state_dict() if "running_" in k], [])


if __name__ == "__main__":
    unittest.main()
