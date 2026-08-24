"""Example L-GATr-slim network config.

It follows the standard weaver network-config interface (``get_model`` / ``get_loss``).

The kinematic scalar features (log pt, log E, ...) are by default taken from
``pf_features`` as defined in the data config; the four-momenta come from ``pf_vectors``
(px, py, pz, energy). Pass ``-o auxiliary_scalars all`` to instead compute the seven
standardized tagging features inside the model from ``pf_vectors`` and prepend them to
``pf_features``; they must then NOT also be listed in ``pf_features``, and ONNX export
additionally requires ``-o momentum_float64 False`` (onnxruntime lacks float64 kernels
for some of the ops used by the kinematic features).

On CUDA, pass ``-o attention_backend varlen`` (torch >= 2.10 native flash-attention
varlen kernel), ``-o attention_backend flash`` (flash-attn package),
``-o attention_backend xformers`` (xformers memory-efficient attention), or
``-o attention_backend flex`` (torch flex_attention) to drop the padding and run
block-diagonal attention over the packed tokens. The first three have no fp32 kernel
and run attention in half precision, costing ~1 pp of accuracy against ``native``;
``flex`` keeps fp32 and matches ``native``. ONNX export requires ``native``.

This file is intentionally NOT named ``test_*`` so pytest does not collect it.
"""

import torch

from weaver.nn.model.LGATrSlim import LGATrSlimTagger


class LGATrSlimTaggerWrapper(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.mod = LGATrSlimTagger(**kwargs)

    def forward(self, points, features, lorentz_vectors, mask):
        return self.mod(features, v=lorentz_vectors, mask=mask)


def get_model(data_config, **kwargs):
    cfg = dict(
        input_dim=len(data_config.input_dicts["pf_features"]),
        num_classes=len(data_config.label_value),
        # defaults follow the tagging-guide `tag_slim` config
        hidden_v_channels=32,
        hidden_s_channels=96,
        num_blocks=12,
        num_heads=8,
        mlp_ratio=4,
        attn_ratio=1,
    )
    cfg.update(**kwargs)
    model = LGATrSlimTaggerWrapper(**cfg)
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
