"""Plain (non-equivariant) transformer jet tagger from the tagging-guide paper.

This is the "Transformer" baseline of

 - "A Guide to Optimal Transformer Architectures for Jet Tagging"
   https://arxiv.org/abs/2608.02735 (tagging-guide, ``model=tr`` reference setup)

i.e. the LLoCa-Transformer backbone (``lloca.backbone.transformer_v2.Transformer``)
run with *identity* frames: no frames-net, no local-frame feature transforms, and the
tensorial attention reduces to standard scaled-dot-product attention. What remains is a
standard pre-norm transformer with RMSNorm, a gated (GLU) GELU MLP, a global class
token, and the seven standardized kinematic tagging features as inputs.

Ported from https://github.com/heidelberg-hepml/tagging-guide: the identity-frames
shortcut of ``experiments/tagging/wrappers.py::TransformerWrapper`` and the embedding in
``experiments/tagging/embedding.py``, with the backbone reused from the existing
:mod:`weaver.nn.model.LLoCaTransformer` port (``lloca/backbone/transformer_v2.py``).

Differences with respect to the upstream implementation:
 - Everything runs on the dense zero-padded (batch, particles, channels) layout used by
   weaver; ``attention_backend="varlen"``, ``"flash"``, or ``"xformers"`` instead drop
   the padding and run block-diagonal attention over the packed tokens (see
   :class:`weaver.nn.model.LGATrSlim` for the backend implementations), mirroring the
   upstream sparse path.
 - The upstream data pipeline prepends the symmetry-breaking spurions at the data level;
   in the identity-frames path they survive as featureless (all-zero) valid tokens. They
   are reproduced here as ``num_register_tokens`` zero-feature tokens (the tagging-guide
   default ``beam_reference=all`` plus time reference gives 4).
 - ONNX export requires ``momentum_float64=False`` (onnxruntime lacks float64 kernels
   for some of the involved ops, e.g. Atan) and the ``"native"`` attention backend.
"""

from __future__ import annotations

import torch
from torch import nn

from weaver.nn.model.LGATrSlim import (
    ATTENTION_BACKENDS,
    dense_to_sparse,
    get_sparse_attention_kwargs,
    insert_global_tokens,
)
from weaver.nn.model.LLoCaTransformer import (
    Frames,
    LLoCaTransformer,
    get_auxiliary_scalars,
    get_num_auxiliary_scalars,
)
from weaver.nn.model.ParticleTransformer import SequenceTrimmer
from weaver.utils.logger import _logger


class PlainTransformerTagger(nn.Module):
    """Weaver-facing plain-transformer jet tagger (tagging-guide ``tr`` baseline).

    Dense (zero-padded) port of the identity-frames path of the tagging-guide
    ``TransformerWrapper``: the seven standardized kinematic tagging features (log pt,
    log E, log pt_rel, log E_rel, dphi, deta, dr) are computed internally from the
    four-momenta, concatenated with the extra scalar features, and fed through a
    pre-norm RMSNorm/GLU transformer. The per-jet logits are read off a global class
    token (or a masked mean over tokens when ``mean_aggregation=True``).

    The weaver data config only needs to provide the extra particle features via
    ``pf_features`` and the four-momenta via ``pf_vectors``; the kinematic features
    must NOT be included in ``pf_features``.

    Parameters
    ----------
    input_dim
        Number of extra scalar features per particle (``pf_features``).
    num_classes
        Number of output classes.
    embed_dim
        Transformer width (must be divisible by ``num_heads``). The tagging-guide
        ``tr`` config at size 0 uses 128.
    num_heads / num_blocks / attention_factor / mlp_factor / dropout_prob /
    elementwise_affine / checkpoint_blocks
        Forwarded to the transformer backbone (defaults follow the tagging-guide
        ``tr`` config at size 0).
    num_register_tokens
        Number of featureless (all-zero) valid tokens prepended to each jet. The
        tagging-guide data pipeline prepends the 4 symmetry-breaking spurions
        (``beam_reference=all`` + time reference), which act as exactly such tokens in
        the identity-frames path; set to 0 to disable.
    auxiliary_scalars
        Which kinematic features to compute: 'all', 'zinvariant', 'so3invariant', or
        None.
    mean_aggregation
        If True, aggregate with a masked mean over tokens instead of a class token.
    momentum_float64
        Whether to compute the kinematic features in float64 (the tagging-guide
        default).
    attention_backend
        ``"native"`` (default) runs on the dense zero-padded layout through
        ``torch.nn.functional.scaled_dot_product_attention``. ``"varlen"`` (torch's
        native flash-attention varlen kernel, torch >= 2.10), ``"flash"`` (the
        flash-attn package), and ``"xformers"`` (memory-efficient attention with a
        block-diagonal mask) drop the padding and run block-diagonal attention over
        the packed tokens instead. These packed backends require CUDA; on CPU the
        packed layout falls back to a materialized block-diagonal SDPA mask. ONNX
        export requires ``"native"``.
    trim
        Whether to enable sequence trimming during training.
    use_amp
        Whether to run the transformer under ``torch.autocast``.
    for_inference
        Whether to apply a softmax to the output (for deployment/ONNX export).
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        # transformer configuration (tagging-guide `tr` config at size 0)
        embed_dim: int = 128,
        num_heads: int = 8,
        num_blocks: int = 8,
        attention_factor: int = 1,
        mlp_factor: int = 2,
        dropout_prob: float | None = None,
        elementwise_affine: bool = True,
        checkpoint_blocks: bool = False,
        # embedding / aggregation
        num_register_tokens: int = 4,
        auxiliary_scalars: str | None = "all",
        mean_aggregation: bool = False,
        momentum_float64: bool = True,
        # attention
        attention_backend: str = "native",
        # misc
        compile_model: bool = False,
        compile_kwargs: dict | None = None,
        trim: bool = True,
        use_amp: bool = False,
        for_inference: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        _logger.info("PlainTransformerTagger init-ed: %s", locals())

        if attention_backend not in ATTENTION_BACKENDS:
            raise ValueError(
                f"Unsupported attention_backend: {attention_backend}. "
                f"Supported backends: {ATTENTION_BACKENDS}."
            )
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim={embed_dim} must be divisible by num_heads={num_heads}")
        self.num_register_tokens = num_register_tokens
        self.auxiliary_scalars = auxiliary_scalars
        self.mean_aggregation = mean_aggregation
        self.momentum_float64 = momentum_float64
        self.attention_backend = attention_backend
        self.use_amp = use_amp
        self.for_inference = for_inference

        # the transformer sees the kinematic features, the extra scalars, and (unless
        # mean-aggregating) the global-token flag channel
        in_channels = (
            get_num_auxiliary_scalars(auxiliary_scalars)
            + input_dim
            + (0 if mean_aggregation else 1)
        )
        # with identity frames the head representation is irrelevant (the reps transform
        # is a no-op and attention is standard SDPA), only the total width matters, so
        # pure scalar heads reproduce the tagging-guide `tr` backbone exactly
        self.net = LLoCaTransformer(
            in_channels=in_channels,
            attn_reps=f"{embed_dim // num_heads}x0n",
            out_channels=num_classes,
            num_blocks=num_blocks,
            num_heads=num_heads,
            checkpoint_blocks=checkpoint_blocks,
            attention_factor=attention_factor,
            mlp_factor=mlp_factor,
            dropout_prob=dropout_prob,
            preserve_variance=False,
            elementwise_affine=elementwise_affine,
            compile=compile_model,
            compile_kwargs=compile_kwargs,
        )

        self.trimmer = SequenceTrimmer(enabled=trim and not for_inference)

    def _identity_frames(self, features: torch.Tensor) -> Frames:
        return Frames(
            is_identity=True,
            device=features.device,
            dtype=features.dtype,
            shape=features.shape[:-1],
        )

    def forward(self, x, v=None, mask=None):
        # x: (N, C, P) -- extra scalar features
        # v: (N, 4, P) [px,py,pz,energy]
        # mask: (N, 1, P) -- real particle = 1, padded = 0
        with torch.no_grad():
            x, v, mask, _ = self.trimmer(x, v, mask)
        mask = mask.squeeze(1).bool()  # (N, P)
        batch_size = x.size(0)

        scalars = x.transpose(1, 2)  # (N, P, C)
        # (E, px, py, pz) convention
        fourmomenta = v.transpose(1, 2)[..., [3, 0, 1, 2]]
        if self.momentum_float64:
            fourmomenta = fourmomenta.to(torch.float64)
        # zero out padded entries (data configs may pad by wrapping real particles)
        fourmomenta = fourmomenta * mask.unsqueeze(-1)
        scalars = scalars * mask.unsqueeze(-1)

        # global-frame kinematic features; zeroed on padding
        jet = fourmomenta.sum(dim=1, keepdim=True)  # (N, 1, 4)
        aux = get_auxiliary_scalars(fourmomenta, jet, auxiliary_scalars=self.auxiliary_scalars)
        aux = aux.to(scalars.dtype) * mask.unsqueeze(-1)
        features = torch.cat([aux, scalars], dim=-1)

        # prepend featureless register tokens (the spurion positions in tagging-guide)
        if self.num_register_tokens:
            features = torch.cat(
                [
                    features.new_zeros(batch_size, self.num_register_tokens, features.size(2)),
                    features,
                ],
                dim=1,
            )
            mask = torch.cat(
                [mask.new_ones(batch_size, self.num_register_tokens), mask], dim=1
            )

        if self.attention_backend != "native":
            return self._forward_packed(features, mask)

        # handle global token: one-hot flag in an extra scalar channel
        if not self.mean_aggregation:
            new_features = features.new_zeros(
                batch_size, features.size(1) + 1, features.size(2) + 1
            )
            new_features[:, 1:, :-1] = features
            new_features[:, 0, -1] = 1.0
            features = new_features
            mask = torch.cat([mask.new_ones(batch_size, 1), mask], dim=1)

        frames = self._identity_frames(features)
        attn_mask = mask.unsqueeze(1).unsqueeze(2)  # (N, 1, 1, tokens)
        with torch.autocast(features.device.type, enabled=self.use_amp):
            outputs = self.net(inputs=features, frames=frames, attn_mask=attn_mask)
        outputs = outputs * mask.unsqueeze(-1)

        # aggregation (the masked mean includes the register tokens, as upstream)
        if self.mean_aggregation:
            output = outputs.sum(dim=-2) / mask.sum(dim=-1, keepdim=True)
        else:
            output = outputs[:, 0]

        if self.for_inference:
            output = torch.softmax(output, dim=1)
        return output

    def _forward_packed(self, features, mask):
        """Packed (sparse) forward path: drop the padding and run block-diagonal varlen
        attention over the concatenated tokens (port of the tagging-guide
        ``TransformerWrapper`` identity-frames sparse path).

        Parameters
        ----------
        features : torch.Tensor
            Features of shape (B, P, C), zeroed on padding.
        mask : torch.BoolTensor
            Valid-token mask of shape (B, P), register tokens included.
        """
        # any upper bound on the per-event sequence lengths works; using the dense width
        # avoids a device-to-host sync
        maxlen = mask.size(1)
        [features], batch, ptr = dense_to_sparse([features], mask)

        if not self.mean_aggregation:
            # prepend a global token per event, one-hot flag in an extra scalar channel
            maxlen = maxlen + 1
            global_idxs, nonglobal_idxs, ptr, batch, num_total = insert_global_tokens(
                ptr, batch, features.shape[0]
            )
            new_features = features.new_zeros(num_total, features.shape[-1] + 1)
            new_features[nonglobal_idxs, :-1] = features
            new_features[:, -1].index_fill_(0, global_idxs, 1.0)
            features = new_features

        attn_kwargs = get_sparse_attention_kwargs(ptr, batch, maxlen, self.attention_backend)

        features = features.unsqueeze(0)  # (1, tokens, C)
        frames = self._identity_frames(features)
        with torch.autocast(features.device.type, enabled=self.use_amp):
            outputs = self.net(inputs=features, frames=frames, **attn_kwargs)
        outputs = outputs.squeeze(0)  # (tokens, num_classes)

        # aggregation
        if self.mean_aggregation:
            batch_size = ptr.numel() - 1
            counts = (ptr[1:] - ptr[:-1]).unsqueeze(-1).to(outputs.dtype)
            output = (
                outputs.new_zeros(batch_size, outputs.shape[-1]).index_add_(0, batch, outputs)
                / counts
            )
        else:
            output = outputs.index_select(0, global_idxs)

        if self.for_inference:
            output = torch.softmax(output, dim=1)
        return output
