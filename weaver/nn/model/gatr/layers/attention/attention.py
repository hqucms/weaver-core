"""Self-attention layers."""

from torch import nn

from gatr.layers.attention.config import SelfAttentionConfig
from gatr.primitives.attention import sdp_attention


class GeometricAttention(nn.Module):
    """Geometric attention layer.

    This is the main attention mechanism used in L-GATr.

    Given multivector and scalar queries, keys, and values, this layer computes:

    ```
    attn_weights[..., i, j] = softmax_j[
        ga_inner_product(q_mv[..., i, :, :], k_mv[..., j, :, :])
        + euclidean_inner_product(q_s[..., i, :], k_s[..., j, :])
    ]
    out_mv[..., i, c, :] = sum_j attn_weights[..., i, j] v_mv[..., j, c, :] / norm
    out_s[..., i, c] = sum_j attn_weights[..., i, j] v_s[..., j, c] / norm
    ```

    Parameters
    ----------
    config : SelfAttentionConfig
        Attention configuration.
    """

    def __init__(self, config: SelfAttentionConfig) -> None:
        super().__init__()

    def forward(self, q_mv, k_mv, v_mv, q_s, k_s, v_s, attention_mask=None):
        """Forward pass through geometric attention.

        Given multivector and scalar queries, keys, and values, this forward pass computes:

        ```
        attn_weights[..., i, j] = softmax_j[
            ga_inner_product(q_mv[..., i, :, :], k_mv[..., j, :, :])
            + euclidean_inner_product(q_s[..., i, :], k_s[..., j, :])
        ]
        out_mv[..., i, c, :] = sum_j attn_weights[..., i, j] v_mv[..., j, c, :] / norm
        out_s[..., i, c] = sum_j attn_weights[..., i, j] v_s[..., j, c] / norm
        ```

        Parameters
        ----------
        q_mv : Tensor with shape (..., num_items_out, num_mv_channels_in, 16)
            Queries, multivector part.
        k_mv : Tensor with shape (..., num_items_in, num_mv_channels_in, 16)
            Keys, multivector part.
        v_mv : Tensor with shape (..., num_items_in, num_mv_channels_out, 16)
            Values, multivector part.
        q_s : Tensor with shape (..., heads, num_items_out, num_s_channels_in)
            Queries, scalar part.
        k_s : Tensor with shape (..., heads, num_items_in, num_s_channels_in)
            Keys, scalar part.
        v_s : Tensor with shape (..., heads, num_items_in, num_s_channels_out)
            Values, scalar part.
        attention_mask: None or Tensor or AttentionBias
            Optional attention mask.
        """

        h_mv, h_s = sdp_attention(
            q_mv,
            k_mv,
            v_mv,
            q_s,
            k_s,
            v_s,
            attn_mask=attention_mask,
        )

        return h_mv, h_s
