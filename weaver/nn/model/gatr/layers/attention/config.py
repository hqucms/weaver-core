from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional


@dataclass
class SelfAttentionConfig:
    """Configuration for attention.

    Parameters
    ----------
    in_mv_channels : int
        Number of input multivector channels.
    out_mv_channels : int
        Number of output multivector channels.
    num_heads : int
        Number of attention heads.
    in_s_channels : int
        Input scalar channels. If None, no scalars are expected nor returned.
    out_s_channels : int
        Output scalar channels. If None, no scalars are expected nor returned.
    additional_qk_mv_channels : int
        Whether additional multivector features for the keys and queries will be provided.
    additional_qk_s_channels : int
        Whether additional scalar features for the keys and queries will be provided.
    multi_query: bool
        Whether to do multi-query attention
    pos_encoding : bool
        Whether to apply rotary positional embeddings along the item dimension to the scalar keys
        and queries.
    pos_encoding_base : int
        Base for the frequencies in the positional encoding.
    output_init : str
        Initialization scheme for final linear layer
    increase_hidden_channels : int
        Factor by which to increase the number of hidden channels (both multivectors and scalars)
    dropout_prob : float or None
        Dropout probability
    head_scale: bool
        Whether to use HeadScaleMHA following the NormFormer (https://arxiv.org/pdf/2110.09456)
    """

    multi_query: bool = True
    in_mv_channels: Optional[int] = None
    out_mv_channels: Optional[int] = None
    in_s_channels: Optional[int] = None
    out_s_channels: Optional[int] = None
    num_heads: int = 8
    additional_qk_mv_channels: int = 0
    additional_qk_s_channels: int = 0
    pos_encoding: bool = False
    pos_encoding_base: int = 4096
    output_init: str = "default"
    checkpoint: bool = True
    increase_hidden_channels: int = 2
    dropout_prob: Optional[float] = None
    head_scale: bool = False

    def __post_init__(self):
        """Type checking / conversion."""
        if isinstance(self.dropout_prob, str) and self.dropout_prob.lower() in [
            "null",
            "none",
        ]:
            self.dropout_prob = None

    @property
    def hidden_mv_channels(self) -> Optional[int]:
        """Returns the number of hidden multivector channels."""

        if self.in_mv_channels is None:
            return None

        return max(
            self.increase_hidden_channels * self.in_mv_channels // self.num_heads, 1
        )

    @property
    def hidden_s_channels(self) -> Optional[int]:
        """Returns the number of hidden scalar channels."""

        if self.in_s_channels is None:
            return None

        hidden_s_channels = max(
            self.increase_hidden_channels * self.in_s_channels // self.num_heads, 4
        )

        # When using positional encoding, the number of scalar hidden channels needs to be even.
        # It also should not be too small.
        if self.pos_encoding:
            hidden_s_channels = (hidden_s_channels + 1) // 2 * 2
            hidden_s_channels = max(hidden_s_channels, 8)

        return hidden_s_channels

    @classmethod
    def cast(cls, config: Any) -> SelfAttentionConfig:
        """Casts an object as SelfAttentionConfig."""
        if isinstance(config, SelfAttentionConfig):
            return config
        if isinstance(config, Mapping):
            return cls(**config)
        raise ValueError(f"Can not cast {config} to {cls}")


@dataclass
class CrossAttentionConfig:
    """Configuration for cross-attention.

    Parameters
    ----------
    in_q_mv_channels : int
        Number of input query multivector channels.
    in_kv_mv_channels : int
        Number of input key/value multivector channels.
    out_mv_channels : int
        Number of output multivector channels.
    num_heads : int
        Number of attention heads.
    in_q_s_channels : int
        Input query scalar channels. If None, no scalars are expected nor returned.
    in_kv_s_channels : int
        Input key/value scalar channels. If None, no scalars are expected nor returned.
    out_s_channels : int
        Output scalar channels. If None, no scalars are expected nor returned.
    additional_q_mv_channels : int
        Whether additional multivector features for the queries will be provided.
    additional_q_s_channels : int
        Whether additional scalar features for the queries will be provided.
    additional_k_mv_channels : int
        Whether additional multivector features for the keys will be provided.
    additional_k_s_channels : int
        Whether additional scalar features for the keys will be provided.
    multi_query: bool
        Whether to do multi-query attention
    output_init : str
        Initialization scheme for final linear layer
    increase_hidden_channels : int
        Factor by which to increase the number of hidden channels (both multivectors and scalars)
    dropout_prob : float or None
        Dropout probability
    head_scale: bool
        Whether to use HeadScaleMHA following the NormFormer (https://arxiv.org/pdf/2110.09456)
    """

    in_q_mv_channels: Optional[int] = None
    in_kv_mv_channels: Optional[int] = None
    out_mv_channels: Optional[int] = None
    out_s_channels: Optional[int] = None
    in_q_s_channels: Optional[int] = None
    in_kv_s_channels: Optional[int] = None
    num_heads: int = 8
    additional_q_mv_channels: int = 0
    additional_q_s_channels: int = 0
    additional_k_mv_channels: int = 0
    additional_k_s_channels: int = 0
    multi_query: bool = True
    output_init: str = "default"
    checkpoint: bool = True
    increase_hidden_channels: int = 2
    dropout_prob: Optional[float] = None
    head_scale: bool = False

    def __post_init__(self):
        """Type checking / conversion."""
        if isinstance(self.dropout_prob, str) and self.dropout_prob.lower() in [
            "null",
            "none",
        ]:
            self.dropout_prob = None

    @property
    def hidden_mv_channels(self) -> Optional[int]:
        """Returns the number of hidden multivector channels."""

        if self.in_q_mv_channels is None:
            return None

        return max(
            self.increase_hidden_channels * self.in_q_mv_channels // self.num_heads, 1
        )

    @property
    def hidden_s_channels(self) -> Optional[int]:
        """Returns the number of hidden scalar channels."""

        if self.in_q_s_channels is None:
            assert self.in_kv_s_channels is None
            return None

        return max(
            self.increase_hidden_channels * self.in_q_s_channels // self.num_heads, 4
        )

    @classmethod
    def cast(cls, config: Any) -> CrossAttentionConfig:
        """Casts an object as CrossAttentionConfig."""
        if isinstance(config, CrossAttentionConfig):
            return config
        if isinstance(config, Mapping):
            return cls(**config)
        raise ValueError(f"Can not cast {config} to {cls}")
