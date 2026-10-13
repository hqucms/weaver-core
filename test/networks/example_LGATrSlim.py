"""Example L-GATr-slim network config.

It follows the standard weaver network-config interface (``get_model`` / ``get_loss``).

The kinematic scalar features (log pt, log E, ...) are by default taken from
``pf_features`` as defined in the data config; the four-momenta come from ``pf_vectors``
(px, py, pz, energy). Pass ``-o auxiliary_scalars all`` to instead compute the seven
standardized tagging features inside the model from ``pf_vectors`` and prepend them to
``pf_features``; they must then NOT also be listed in ``pf_features``, and ONNX export
additionally requires ``-o momentum_float64 False`` (onnxruntime lacks float64 kernels
for some of the ops used by the kinematic features).

The default ``attention_backend="auto"`` drops the padding on CUDA and runs
block-diagonal attention over the packed tokens with the fastest kernel for the
precision: ``efficient`` (exact fp32, torch's memory-efficient kernel) without
``--use-amp``, ``varlen`` (torch >= 2.10 flash-attention varlen kernel) under fp16/bf16
AMP. On CPU and for ONNX export it uses ``native`` (dense SDPA). A backend can still be
forced with ``-o attention_backend "'<name>'"``: ``native``, ``varlen``, ``efficient``,
``flex`` (torch flex_attention), ``flash`` (flash-attn package) or ``xformers``.

The four-momenta are mapped to per-jet light-cone coordinates by default
(``vector_coord="lightcone"``): the same function, but well-conditioned, so under
``--use-amp`` attention and the vector GEMMs run in the AMP dtype at fp32 accuracy
(``-o attention_backend flex --use-amp`` is ~2.3x faster than fp32-pinned attention).
``-o vector_coord "'cartesian'"`` restores Cartesian coordinates, where half-precision
attention costs ~1 pp, so train those without ``--use-amp``.

That default axis is the summed constituent momentum, which assumes the input is one
jet. On **full events** it points along the beam, so fit the axis to the event instead:

    -o lightcone_frame "'principal'" -o attention_backend "'varlen'" \
    --use-amp --amp-dtype fp16 \
    --compiler-option options "{'loop_ordering_after_fusion': False}"

That is the accuracy-first *and* speed-first setting: on 10-epoch hh4b runs at four seeds it
is -0.01 pp from Cartesian fp32 at **3.5x** the speed. **Use fp16, not bf16** -- fp16 beat
bf16 on 12 of 12 paired runs by +0.24..+0.27 pp at identical step times, because bf16's
8-bit mantissa, not the packed layout, is what half-precision attention loses to here.
``-o lightcone_frame "'sum'"`` (the default) and ``"'collinear'"`` are within 0.02 pp of
``principal``; Cartesian at the same precision is -0.15 pp behind all three, which is what
the light-cone frame is actually worth.

Precision is the trainer's: everything but the norms and the gate contractions follows
``--use-amp``, and the model takes no precision options of its own.

``-o gate_norm "'euclidean'"`` makes the vector gates scale-free: their argument becomes a
learned per-channel scale times ``<a, b> / (||a|| ||b||)``, which lies in [-1, 1], so a gate
cannot saturate just because the vector stream grew (``-o gate_scale_init`` sets the initial
scale, default 4).

This file is intentionally NOT named ``test_*`` so pytest does not collect it.
"""

import torch

from weaver.nn.model.LGATrSlim import LGATrSlimTagger


class LGATrSlimTaggerWrapper(torch.nn.Module):
    def __init__(self, max_particles=None, **kwargs):
        super().__init__()
        self.mod = LGATrSlimTagger(**kwargs)
        # upper bound on the particles per event, for the unpadded (--jagged nested) input
        self.max_particles = max_particles

    def forward(self, points, features, lorentz_vectors, mask):
        if features.is_nested:
            # (B, j, C) jagged-layout nested tensors: hand the packed values straight to the
            # model, which never builds the padded layout
            return self.mod.forward_jagged(
                features.values(), lorentz_vectors.values(), features.offsets(), self.max_particles
            )
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
        max_particles=data_config.input_shapes["pf_features"][2],
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
