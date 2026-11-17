"""Regression checks for particle trimming and label alignment."""

import unittest
from unittest.mock import patch

import torch

from weaver.nn.model.ParticleNeXt import ParticleNeXt
from weaver.nn.model.ParticleNet import ParticleNet
from weaver.nn.model.ParticleTransformer import (
    ParticleTransformer,
    ParticleTransformerTagger,
    ParticleTransformerTaggerWithExtraPairFeatures,
    SequenceTrimmer,
)


_PART_CONFIG = dict(
    num_classes=3,
    embed_dims=(8,),
    pair_embed_dims=None,
    num_heads=2,
    num_layers=1,
    num_cls_layers=0,
    fix_init=False,
)
_PN_CONFIG = dict(
    num_classes=3,
    conv_params=[(2, (8,)), (3, (8,))],
    fc_params=[(8, 0.)],
    use_fusion=False,
)
_NEXT_CONFIG = dict(
    feature_input_dim=2,
    num_classes=3,
    node_dim=4,
    edge_dim=0,
    layer_params=[(2, 8)],
    fc_params=[(8, 0.)],
    global_aggregation="mean",
    use_rel_lv_fts=False,
    use_node_se=False,
    use_edge_se=False,
)


def _inputs(mask_values):
    mask = torch.tensor(mask_values, dtype=torch.bool).unsqueeze(1)
    batch, _, length = mask.shape
    return torch.randn(batch, 2, length), torch.randn(batch, 2, length), mask


class SequenceTrimmingTest(unittest.TestCase):
    def test_eval_rejects_trimming_valid_particles(self):
        for masks in ([[0, 1, 1, 0]], [[1, 1, 0, 0], [0, 0, 1, 0]]):
            with self.subTest(masks=masks):
                x, _, mask = _inputs(masks)
                trimmer = SequenceTrimmer(enabled=True, warmup_steps=0).eval()
                with self.assertRaisesRegex(RuntimeError, "removed valid particles.*trim=False"):
                    trimmer(x, mask=mask)

    def test_eval_preserves_all_valid_particles(self):
        for masks in ([[1, 1, 0, 0], [1, 0, 0, 0]], [[1, 1, 0, 0], [0, 1, 0, 0]]):
            with self.subTest(masks=masks):
                x, _, mask = _inputs(masks)
                trimmer = SequenceTrimmer(enabled=True, warmup_steps=0).eval()
                output, _, output_mask, _ = trimmer(x, mask=mask)
                self.assertEqual(output.size(-1), 2)
                torch.testing.assert_close(output_mask.sum(-1), mask.sum(-1))


class ParticleNeXtTrimmingTest(unittest.TestCase):
    def test_eval_rejects_trimming_valid_particles(self):
        model = ParticleNeXt(**_NEXT_CONFIG).eval()
        model._counter = 5
        points, features, mask = _inputs([[0, 1, 0, 1, 0, 1, 0, 1]])
        vectors = torch.randn(1, 4, 8)
        with self.assertRaisesRegex(RuntimeError, "removed valid particles.*trim=False"):
            model(points, features, vectors, mask)

    def test_eval_allows_holes_when_no_valid_particles_are_lost(self):
        model = ParticleNeXt(**_NEXT_CONFIG).eval()
        model._counter = 5
        points, features, mask = _inputs([[1, 1, 0, 0], [0, 1, 0, 0]])
        seen = []

        def capture(module, args, kwargs):
            seen.append(kwargs["mask"].clone())

        handle = model.layers[0].register_forward_pre_hook(capture, with_kwargs=True)
        with torch.no_grad():
            output = model(points, features, torch.randn(2, 4, 4), mask)
        handle.remove()
        self.assertTrue(output.isfinite().all())
        self.assertEqual(seen[0].size(-1), 2)
        torch.testing.assert_close(seen[0].sum(-1), mask.sum(-1))

    def test_training_can_still_drop_valid_particles(self):
        model = ParticleNeXt(**_NEXT_CONFIG).train()
        model._counter = 5
        points, features, mask = _inputs([[1, 1, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1]])
        with patch("weaver.nn.model.ParticleNeXt.random.uniform", return_value=0.5), torch.no_grad():
            output = model(points, features, torch.randn(2, 4, 6), mask)
        self.assertEqual(output.shape, (2, 3))
        self.assertTrue(output.isfinite().all())


class SegmentationTrimmingConfigTest(unittest.TestCase):
    def models(self):
        return (
            (ParticleTransformer, dict(input_dim=2, **_PART_CONFIG)),
            (ParticleTransformerTagger, dict(pf_input_dim=2, sv_input_dim=2, **_PART_CONFIG)),
            (ParticleTransformerTaggerWithExtraPairFeatures, dict(pf_input_dim=2, sv_input_dim=2, **_PART_CONFIG)),
            (ParticleNet, dict(input_dims=2, **_PN_CONFIG)),
            (ParticleNeXt, _NEXT_CONFIG),
        )

    def test_segmentation_rejects_internal_trimming_before_forward(self):
        for model_class, config in self.models():
            with self.subTest(model=model_class.__name__):
                with self.assertRaisesRegex(ValueError, "trim=False.*inputs, masks, and labels.*one SequenceTrimmer call"):
                    model_class(**config, for_segmentation=True, trim=True)

    def test_segmentation_accepts_disabled_or_inference_trimming(self):
        for model_class, config in self.models():
            for flags in (dict(trim=False), dict(trim=True, for_inference=True)):
                with self.subTest(model=model_class.__name__, flags=flags):
                    model = model_class(**config, for_segmentation=True, **flags)
                    for module in model.modules():
                        if isinstance(module, SequenceTrimmer):
                            self.assertFalse(module.enabled)
                    if isinstance(model, ParticleNeXt):
                        self.assertFalse(model.trim and not model.for_inference)


if __name__ == "__main__":
    unittest.main()
