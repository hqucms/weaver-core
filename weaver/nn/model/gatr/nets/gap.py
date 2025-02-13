from dataclasses import replace
from typing import Optional, Tuple, Union

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from gatr.layers.linear import EquiLinear
from gatr.layers.mlp import MLPConfig, GeoMLP


class GAP(nn.Module):
    """Geometric Algebra Perceptron network for a data with a single token dimension.
    It combines `num_blocks` GeoMLP blocks.

    Assumes input has shape `(..., in_channels, 16)`, output has shape
    `(..., out_channels, 16)`, will create hidden representations with shape
    `(..., hidden_channels, 16)`.

    Parameters
    ----------
    in_mv_channels : int
        Number of input multivector channels.
    out_mv_channels : int
        Number of output multivector channels.
    hidden_mv_channels : int
        Number of hidden multivector channels.
    in_s_channels : None or int
        If not None, sets the number of scalar input channels.
    out_s_channels : None or int
        If not None, sets the number of scalar output channels.
    hidden_s_channels : None or int
        If not None, sets the number of scalar hidden channels.
    num_blocks : int
        Number of resnet blocks.
    dropout_prob : float or None
        Dropout probability
    """

    def __init__(
        self,
        in_mv_channels: int,
        out_mv_channels: int,
        hidden_mv_channels: int,
        in_s_channels: Optional[int],
        out_s_channels: Optional[int],
        hidden_s_channels: Optional[int],
        mlp: MLPConfig,
        num_blocks: int = 10,
        num_layers: int = 3,
        checkpoint_blocks: bool = False,
        dropout_prob: Optional[float] = None,
        **kwargs,
    ) -> None:
        super().__init__()

        self.linear_in = EquiLinear(
            in_mv_channels,
            hidden_mv_channels,
            in_s_channels=in_s_channels,
            out_s_channels=hidden_s_channels,
        )

        mlp = MLPConfig.cast(mlp)
        mlp = replace(
            mlp,
            mv_channels=[hidden_mv_channels for _ in range(num_layers)],
            s_channels=[hidden_s_channels for _ in range(num_layers)],
            dropout_prob=dropout_prob,
        )
        self.blocks = nn.ModuleList([GeoMLP(mlp) for _ in range(num_blocks)])

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
        scalars: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Union[torch.Tensor, None]]:
        """Forward pass of the network.

        Parameters
        ----------
        multivectors : torch.Tensor with shape (..., in_mv_channels, 16)
            Input multivectors.
        scalars : None or torch.Tensor with shape (..., in_s_channels)
            Optional input scalars.

        Returns
        -------
        outputs_mv : torch.Tensor with shape (..., out_mv_channels, 16)
            Output multivectors.
        outputs_s : None or torch.Tensor with shape (..., out_s_channels)
            Output scalars, if scalars are provided. Otherwise None.
        """

        # Pass through the blocks
        h_mv, h_s = self.linear_in(multivectors, scalars=scalars)
        for block in self.blocks:
            if self._checkpoint_blocks:
                h_mv, h_s = checkpoint(
                    block,
                    h_mv,
                    use_reentrant=False,
                    scalars=h_s,
                )
            else:
                h_mv, h_s = block(
                    h_mv,
                    scalars=h_s,
                )
        outputs_mv, outputs_s = self.linear_out(h_mv, scalars=h_s)

        return outputs_mv, outputs_s
