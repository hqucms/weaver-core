from dataclasses import replace
from typing import Optional, Tuple

import torch
from torch import nn

from gatr.layers import (
    SelfAttention,
    CrossAttention,
    SelfAttentionConfig,
    CrossAttentionConfig,
)
from gatr.layers.layer_norm import EquiLayerNorm
from gatr.layers.mlp.config import MLPConfig
from gatr.layers.mlp.mlp import GeoMLP


class ConditionalGATrBlock(nn.Module):
    """Equivariant transformer decoder block for L-GATr.

    Inputs are first processed by a block consisting of LayerNorm, multi-head geometric
    self-attention, and a residual connection. Then the conditions are included with
    cross-attention using the same overhead as in the self-attention part.
    Then the data is processed by a block consisting of
    another LayerNorm, an item-wise two-layer geometric MLP with GeLU activations, and another
    residual connection.

    Parameters
    ----------
    mv_channels : int
        Number of input and output multivector channels
    s_channels: int
        Number of input and output scalar channels
    condition_mv_channels: int
        Number of condition multivector channels
    condition_s_channels: int
        Number of condition scalar channels
    attention: SelfAttentionConfig
        Attention configuration
    crossattention: CrossAttentionConfig
        Cross-attention configuration
    mlp: MLPConfig
        MLP configuration
    dropout_prob : float or None
        Dropout probability
    double_layernorm : bool
        Whether to use double layer normalization
    """

    def __init__(
        self,
        mv_channels: int,
        s_channels: int,
        condition_mv_channels: int,
        condition_s_channels: int,
        attention: SelfAttentionConfig,
        crossattention: CrossAttentionConfig,
        mlp: MLPConfig,
        dropout_prob: Optional[float] = None,
        double_layernorm: bool = False,
    ) -> None:
        super().__init__()

        # Normalization layer (stateless, so we can use the same layer for both normalization
        # instances)
        self.norm = EquiLayerNorm()
        self.double_layernorm = double_layernorm

        # Self-attention layer
        attention = replace(
            attention,
            in_mv_channels=mv_channels,
            out_mv_channels=mv_channels,
            in_s_channels=s_channels,
            out_s_channels=s_channels,
            output_init="small",
            dropout_prob=dropout_prob,
        )
        self.attention = SelfAttention(attention)

        # Cross-attention layer
        crossattention = replace(
            crossattention,
            in_q_mv_channels=mv_channels,
            in_q_s_channels=s_channels,
            in_kv_mv_channels=condition_mv_channels,
            in_kv_s_channels=condition_s_channels,
            out_mv_channels=mv_channels,
            out_s_channels=s_channels,
            output_init="small",
            dropout_prob=dropout_prob,
        )
        self.crossattention = CrossAttention(crossattention)

        # MLP block
        mlp = replace(
            mlp,
            mv_channels=(mv_channels, 2 * mv_channels, mv_channels),
            s_channels=(s_channels, 2 * s_channels, s_channels),
            dropout_prob=dropout_prob,
        )
        self.mlp = GeoMLP(mlp)

    def forward(
        self,
        multivectors: torch.Tensor,
        multivectors_condition: torch.Tensor,
        scalars: torch.Tensor = None,
        scalars_condition: torch.Tensor = None,
        attention_mask=None,
        crossattention_mask=None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the transformer decoder block.

        Parameters
        ----------
        multivectors : torch.Tensor with shape (..., items, channels, 16)
            Input multivectors.
        scalars : torch.Tensor with shape (..., s_channels)
            Input scalars.
        multivectors_condition : torch.Tensor with shape (..., items, channels, 16)
            Input condition multivectors.
        scalars_condition : torch.Tensor with shape (..., s_channels)
            Input condition scalars.
        attention_mask: None or torch.Tensor or AttentionBias
            Optional attention mask.
        crossattention_mask: None or torch.Tensor or AttentionBias
            Optional attention mask for the condition.

        Returns
        -------
        outputs_mv : torch.Tensor with shape (..., items, channels, 16).
            Output multivectors
        output_scalars : torch.Tensor with shape (..., s_channels)
            Output scalars
        """

        # Self-attention block: pre layer norm
        h_mv, h_s = self.norm(multivectors, scalars=scalars)

        # Self-attention block: self attention
        h_mv, h_s = self.attention(
            h_mv,
            scalars=h_s,
            attention_mask=attention_mask,
        )

        # Self-attention block: post layer norm
        if self.double_layernorm:
            h_mv, h_s = self.norm(h_mv, scalars=h_s)

        # Self-attention block: skip connection
        multivectors = multivectors + h_mv
        scalars = scalars + h_s

        # Cross-attention block: pre layer norm
        h_mv, h_s = self.norm(multivectors, scalars=scalars)
        c_mv, c_s = self.norm(multivectors_condition, scalars=scalars_condition)

        # Cross-attention block: cross attention
        h_mv, h_s = self.crossattention(
            multivectors_q=h_mv,
            multivectors_kv=c_mv,
            scalars_q=h_s,
            scalars_kv=c_s,
            attention_mask=crossattention_mask,
        )

        # Cross-attention block: post layer norm
        if self.double_layernorm:
            h_mv, h_s = self.norm(h_mv, scalars=h_s)

        # Cross-attention block: skip connection
        outputs_mv = multivectors + h_mv
        outputs_s = scalars + h_s

        # MLP block: pre layer norm
        h_mv, h_s = self.norm(outputs_mv, scalars=outputs_s)

        # MLP block: MLP
        h_mv, h_s = self.mlp(h_mv, scalars=h_s)

        # MLP block: post layer norm
        if self.double_layernorm:
            h_mv, h_s = self.norm(h_mv, scalars=h_s)

        # MLP block: skip connection
        outputs_mv = outputs_mv + h_mv
        outputs_s = outputs_s + h_s

        return outputs_mv, outputs_s
