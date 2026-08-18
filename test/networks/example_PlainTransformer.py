"""Example plain-transformer network config (tagging-guide ``tr`` baseline).

It follows the standard weaver network-config interface (``get_model`` / ``get_loss``).

The extra particle features are taken from ``pf_features`` as defined in the data
config; the four-momenta come from ``pf_vectors`` (px, py, pz, energy). The seven
kinematic tagging features (log pt, log E, ...) are computed inside the model, so they
must NOT be included in ``pf_features``.

On CUDA, pass ``-o attention_backend varlen`` (torch >= 2.10 native flash-attention
varlen kernel), ``-o attention_backend flash`` (flash-attn package), or
``-o attention_backend xformers`` (xformers memory-efficient attention) to drop the
padding and run block-diagonal attention over the packed tokens.

For ONNX export, pass ``-o momentum_float64 False``: onnxruntime lacks float64 kernels
for some of the involved ops. ONNX export also requires the default ``native``
attention backend.

This file is intentionally NOT named ``test_*`` so pytest does not collect it.
"""

import torch

from weaver.nn.model.PlainTransformer import PlainTransformerTagger


class PlainTransformerTaggerWrapper(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.mod = PlainTransformerTagger(**kwargs)

    def forward(self, points, features, lorentz_vectors, mask):
        return self.mod(features, v=lorentz_vectors, mask=mask)


def get_model(data_config, **kwargs):
    cfg = dict(
        input_dim=len(data_config.input_dicts["pf_features"]),
        num_classes=len(data_config.label_value),
        # defaults follow the tagging-guide `tr` config at size 0
        embed_dim=128,
        num_heads=8,
        num_blocks=8,
        mlp_factor=2,
    )
    cfg.update(**kwargs)
    model = PlainTransformerTaggerWrapper(**cfg)
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
