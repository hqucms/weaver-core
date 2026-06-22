"""Example ParticleTransformer network config.

It follows the standard weaver network-config interface (``get_model`` / ``get_loss``).

This file is intentionally NOT named ``test_*`` so pytest does not collect it.
"""

import torch

from weaver.nn.model.ParticleTransformer import ParticleTransformer


class ParticleTransformerWrapper(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.mod = ParticleTransformer(**kwargs)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"mod.cls_token"}

    def forward(self, points, features, lorentz_vectors, mask):
        return self.mod(features, v=lorentz_vectors, mask=mask)


def get_model(data_config, **kwargs):
    cfg = dict(
        input_dim=len(data_config.input_dicts["pf_features"]),
        num_classes=len(data_config.label_value),
    )
    cfg.update(**kwargs)
    model = ParticleTransformerWrapper(**cfg)
    model_info = {
        "input_names": list(data_config.input_names),
        "input_shapes": {k: ((1,) + s[1:]) for k, s in data_config.input_shapes.items()},
        "output_names": ["softmax"],
        "dynamic_axes": {
            **{k: {0: "N", 2: "n_" + k.split("_")[0]} for k in data_config.input_names},
            **{"softmax": {0: "N"}},
        },
    }
    return model, model_info


def get_loss(data_config, **kwargs):
    return torch.nn.CrossEntropyLoss()
