"""Equivariant transformer for multivector data."""

from dataclasses import replace
from typing import Optional, Tuple, Union

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from gatr.layers import (
    CrossAttentionConfig,
    SelfAttentionConfig,
    GATrBlock,
    ConditionalGATrBlock,
    EquiLinear,
)
from gatr.layers.mlp.config import MLPConfig


class ConditionalGATr(nn.Module):
    """L-GATr network for a data with a single token dimension.

    It combines `num_blocks` L-GATr transformer blocks, each consisting of geometric self-attention
    layers, a geometric MLP, residual connections, and normalization layers. In addition, there
    are initial and final equivariant linear layers.

    Assumes input has shape `(..., items, in_channels, 16)`, output has shape
    `(..., items, out_channels, 16)`, will create hidden representations with shape
    `(..., items, hidden_channels, 16)`.

    Parameters
    ----------
    in_mv_channels : int
        Number of input multivector channels.
    condition_mv_channels : int
        Number of condition multivector channels.
    out_mv_channels : int
        Number of output multivector channels.
    hidden_mv_channels : int
        Number of hidden multivector channels.
    in_s_channels : None or int
        If not None, sets the number of scalar input channels.
    condition_s_channels : None or int
        If not None, sets the number of scalar condition channels.
    out_s_channels : None or int
        If not None, sets the number of scalar output channels.
    hidden_s_channels : None or int
        If not None, sets the number of scalar hidden channels.
    attention: Dict
        Data for SelfAttentionConfig
    crossattention: Dict
        Data for CrossAttentionConfig
    attention_condition: Dict
        Data for SelfAttentionConfig
    mlp: Dict
        Data for MLPConfig
    num_blocks : int
        Number of transformer blocks.
    dropout_prob : float or None
        Dropout probability
    double_layernorm : bool
        Whether to use double layer normalization
    """

    def __init__(
        self,
        in_mv_channels: int,
        condition_mv_channels: int,
        out_mv_channels: int,
        hidden_mv_channels: int,
        in_s_channels: Optional[int],
        condition_s_channels: Optional[int],
        out_s_channels: Optional[int],
        hidden_s_channels: Optional[int],
        attention: SelfAttentionConfig,
        crossattention: CrossAttentionConfig,
        attention_condition: SelfAttentionConfig,
        mlp: MLPConfig,
        num_blocks: int = 10,
        checkpoint_blocks: bool = False,
        dropout_prob: Optional[float] = None,
        double_layernorm: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        self.linear_in_condition = EquiLinear(
            condition_mv_channels,
            hidden_mv_channels,
            in_s_channels=condition_s_channels,
            out_s_channels=hidden_s_channels,
        )
        self.linear_in = EquiLinear(
            in_mv_channels,
            hidden_mv_channels,
            in_s_channels=in_s_channels,
            out_s_channels=hidden_s_channels,
        )
        self.linear_condition = EquiLinear(
            condition_mv_channels,
            hidden_mv_channels,
            in_s_channels=condition_s_channels,
            out_s_channels=hidden_s_channels,
        )
        attention = SelfAttentionConfig.cast(attention)
        crossattention = CrossAttentionConfig.cast(crossattention)
        attention_condition = SelfAttentionConfig.cast(attention_condition)
        mlp = MLPConfig.cast(mlp)
        self.condition_blocks = nn.ModuleList(
            [
                GATrBlock(
                    mv_channels=hidden_mv_channels,
                    s_channels=hidden_s_channels,
                    attention=attention_condition,
                    mlp=mlp,
                    dropout_prob=dropout_prob,
                    double_layernorm=double_layernorm,
                )
                for _ in range(num_blocks)
            ]
        )
        self.blocks = nn.ModuleList(
            [
                ConditionalGATrBlock(
                    mv_channels=hidden_mv_channels,
                    s_channels=hidden_s_channels,
                    condition_mv_channels=hidden_mv_channels,
                    condition_s_channels=hidden_s_channels,
                    attention=attention,
                    crossattention=crossattention,
                    mlp=mlp,
                    dropout_prob=dropout_prob,
                    double_layernorm=double_layernorm,
                )
            ]
        )
        self.linear_out = EquiLinear(
            hidden_mv_channels,
            out_mv_channels,
            in_s_channels=hidden_s_channels,
            out_s_channels=out_s_channels,
        )
        self._checkpoint_blocks = checkpoint_blocks

    def forward(
        self,
        multivectors: torch.Tensor,
        multivectors_condition: torch.Tensor,
        scalars: Optional[torch.Tensor] = None,
        scalars_condition: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        attention_mask_condition: Optional[torch.Tensor] = None,
        crossattention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Union[torch.Tensor, None]]:
        """Forward pass of the network.

        Parameters
        ----------
        multivectors : torch.Tensor with shape (..., num_items, in_mv_channels, 16)
            Input multivectors.
        multivectors_condition : torch.Tensor with shape (..., num_items_condition, in_mv_channels, 16)
            Input multivectors.
        scalars : None or torch.Tensor with shape (..., num_items, in_s_channels)
            Optional input scalars.
        scalars_condition : None or torch.Tensor with shape (..., num_items_condition, in_s_channels)
            Optional input scalars.
        attention_mask: None or torch.Tensor with shape (..., num_items, num_items)
            Optional attention mask
        attention_mask_condition: None or torch.Tensor with shape (..., num_items_condition, num_items_condition)
            Optional attention mask for condition
        crossattention_mask: None or torch.Tensor with shape (..., num_items, num_items_condition)
            Optional mask for cross-attention

        Returns
        -------
        outputs_mv : torch.Tensor with shape (..., num_items, out_mv_channels, 16)
            Output multivectors.
        outputs_s : None or torch.Tensor with shape (..., num_items, out_s_channels)
            Output scalars, if scalars are provided. Otherwise None.
        """

        # Encode condition with GATr blocks
        c_mv, c_s = self.linear_in_condition(
            multivectors_condition, scalars=scalars_condition
        )
        for block in self.condition_blocks:
            if self._checkpoint_blocks:
                c_mv, c_s = checkpoint(
                    block,
                    c_mv,
                    use_reentrant=False,
                    scalars=c_s,
                    attention_mask=attention_mask_condition,
                )
            else:
                c_mv, c_s = block(
                    c_mv,
                    scalars=c_s,
                    attention_mask=attention_mask_condition,
                )

        # Decode condition into main track with
        h_mv, h_s = self.linear_in(multivectors, scalars=scalars)
        for block in self.blocks:
            if self._checkpoint_blocks:
                h_mv, h_s = checkpoint(
                    block,
                    h_mv,
                    use_reentrant=False,
                    scalars=h_s,
                    multivectors_condition=c_mv,
                    scalars_condition=c_s,
                    attention_mask=attention_mask,
                    crossattention_mask=crossattention_mask,
                )
            else:
                h_mv, h_s = block(
                    h_mv,
                    scalars=h_s,
                    multivectors_condition=c_mv,
                    scalars_condition=c_s,
                    attention_mask=attention_mask,
                    crossattention_mask=crossattention_mask,
                )

        outputs_mv, outputs_s = self.linear_out(h_mv, scalars=h_s)

        return outputs_mv, outputs_s
