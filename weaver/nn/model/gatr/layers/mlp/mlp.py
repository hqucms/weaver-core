"""Factory functions for simple MLPs for multivector data."""

from typing import List, Tuple, Union

import torch
from torch import nn

from gatr.layers.dropout import GradeDropout
from gatr.layers.linear import EquiLinear
from gatr.layers.mlp.config import MLPConfig
from gatr.layers.mlp.geometric_bilinears import GeometricBilinear
from gatr.layers.mlp.nonlinearities import ScalarGatedNonlinearity

USE_GEOMETRIC_PRODUCT = True


class GeoMLP(nn.Module):
    """Geometric MLP.

    This is a core component of GATr's transformer blocks. It is similar to a regular MLP, except
    that it uses geometric bilinears (the geometric product) in place of the first linear layer.

    Assumes input has shape `(..., channels[0], 16)`, output has shape `(..., channels[-1], 16)`,
    will create hidden layers with shape `(..., channel, 16)` for each additional entry in
    `channels`.

    Parameters
    ----------
    config: MLPConfig
        Configuration object
    """

    def __init__(
        self,
        config: MLPConfig,
    ) -> None:
        super().__init__()

        # Store settings
        self.config = config

        assert config.mv_channels is not None
        s_channels = (
            [None for _ in config.mv_channels]
            if config.s_channels is None
            else config.s_channels
        )

        layers: List[nn.Module] = []

        if len(config.mv_channels) >= 2:
            kwargs = dict(
                in_mv_channels=config.mv_channels[0],
                out_mv_channels=config.mv_channels[1],
                in_s_channels=s_channels[0],
                out_s_channels=s_channels[1],
            )
            if USE_GEOMETRIC_PRODUCT:
                layers.append(GeometricBilinear(**kwargs))
            else:
                layers.append(ScalarGatedNonlinearity(config.activation))
                layers.append(EquiLinear(**kwargs))
            if config.dropout_prob is not None:
                layers.append(GradeDropout(config.dropout_prob))

            for in_, out, in_s, out_s in zip(
                config.mv_channels[1:-1],
                config.mv_channels[2:],
                s_channels[1:-1],
                s_channels[2:],
            ):
                layers.append(ScalarGatedNonlinearity(config.activation))
                layers.append(
                    EquiLinear(in_, out, in_s_channels=in_s, out_s_channels=out_s)
                )
                if config.dropout_prob is not None:
                    layers.append(GradeDropout(config.dropout_prob))

        self.layers = nn.ModuleList(layers)

    def forward(
        self, multivectors: torch.Tensor, scalars: torch.Tensor
    ) -> Tuple[torch.Tensor, Union[torch.Tensor, None]]:
        """Forward pass.

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

        mv, s = multivectors, scalars

        for i, layer in enumerate(self.layers):
            mv, s = layer(mv, scalars=s)

        return mv, s
