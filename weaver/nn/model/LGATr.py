import torch
from torch import nn

from .gatr import GATr, SelfAttentionConfig, MLPConfig
from .gatr.interface import (
    embed_vector,
    extract_scalar,
    get_num_spurions,
    embed_spurions,
)
from .ParticleTransformer import SequenceTrimmer


class LGATrWrapper(nn.Module):
    """
    Wrapper that handles interface to the GATr code
    - create dataclasses for attention and mlp
    - append spurions (symmetry-breaking)
    - interface to geometric algebra
    - mean aggregation to extract tagging score
    """

    def __init__(
        self,
        in_s_channels,
        hidden_mv_channels,
        hidden_s_channels,
        num_classes,
        num_blocks,
        num_heads,
        global_token=True,
        spurion_token=True,
        beam_reference="xyplane",
        add_time_reference=True,
        two_beams=True,
        activation="gelu",
        multi_query=False,
        increase_hidden_channels=2,
        head_scale=False,
        double_layernorm=False,
        dropout_prob=None,
        checkpoint_blocks=False,
    ):
        super().__init__()

        # spurion business
        in_mv_channels = 1
        self.global_token = global_token
        self.spurion_token = spurion_token

        num_spurions = get_num_spurions(
            beam_reference, add_time_reference, two_beams=two_beams
        )
        if not self.spurion_token:
            in_mv_channels += num_spurions
        self.spurion_kwargs = {
            "beam_reference": beam_reference,
            "add_time_reference": add_time_reference,
            "two_beams": two_beams,
        }

        attention = SelfAttentionConfig(
            multi_query=multi_query,
            num_heads=num_heads,
            increase_hidden_channels=increase_hidden_channels,
            dropout_prob=dropout_prob,
            head_scale=head_scale,
        )
        mlp = MLPConfig(
            activation=activation,
            dropout_prob=dropout_prob,
        )

        self.net = GATr(
            in_mv_channels=in_mv_channels,
            out_mv_channels=num_classes,
            hidden_mv_channels=hidden_mv_channels,
            in_s_channels=in_s_channels,
            out_s_channels=None,
            hidden_s_channels=hidden_s_channels,
            attention=attention,
            mlp=mlp,
            num_blocks=num_blocks,
            double_layernorm=double_layernorm,
            dropout_prob=dropout_prob,
            checkpoint_blocks=checkpoint_blocks,
        )

    def forward(self, x, v, mask):
        # reshape input
        x = x.transpose(1, 2)  # (batch_size, seq_len, num_fts)
        v = v.transpose(1, 2)  # (batch_size, seq_len, 4)
        mask = mask.transpose(1, 2)  # (batchsize, seq_len, 1)

        # geometric algebra embedding
        fourmomenta = v[:, :, None, [3, 0, 1, 2]]  # (px, py, pz, E) -> (E, px, py, pz)
        mv = embed_vector(fourmomenta)
        s = x

        # spurion business
        spurions = embed_spurions(**self.spurion_kwargs).to(
            device=s.device, dtype=s.dtype
        )
        if self.spurion_token:
            # prepend spurions as extra tokens
            # (have to also extend mask and zero-pad scalars)
            mask_ones = torch.ones(
                mask.shape[0],
                spurions.shape[0],
                mask.shape[2],
                device=mask.device,
                dtype=mask.dtype,
            )
            mask = torch.cat([mask_ones, mask], dim=1)

            s_zeros = torch.zeros(
                s.shape[0],
                spurions.shape[0],
                s.shape[2],
                device=s.device,
                dtype=s.dtype,
            )
            s = torch.cat([s_zeros, s], dim=1)

            spurions = spurions[None, :, None, :].repeat(mv.shape[0], 1, 1, 1)
            mv = torch.cat([spurions, mv], dim=1)
        else:
            # add spurions as extra mv channels
            spurions = spurions[None, None, :, :].repeat(mv.shape[0], mv.shape[1], 1, 1)
            mv = torch.cat([mv, spurions], dim=2)

        # global token business
        if self.global_token:
            # prepend global token as first particle in the list
            global_token = torch.zeros_like(mv[:, [0], :, :])
            mv = torch.cat((global_token, mv), dim=1)
            mask_ones = torch.ones_like(mask[:, [0]])
            mask = torch.cat((mask_ones, mask), dim=1)
            s_zeros = torch.zeros_like(s[:, [0]])
            s = torch.cat((s_zeros, s), dim=1)

        # reshape mask to broadcast correctly
        mask = mask.bool()
        mask = mask[:, None, None, :, 0]  # (batch_size, 1, 1, seq_len)

        # call network
        out_mv, _ = self.net(mv, s, mask)
        output = extract_scalar(out_mv)[..., 0]

        # aggregation
        if self.global_token:
            output = output[:, 0]
        else:
            # mean aggregation
            output[~mask[:, 0, 0]] = 0.0
            output = output.mean(dim=1)
        return output


class LGATrTagger(nn.Module):
    """Mimic weaver features"""

    def __init__(
        self,
        use_amp=False,
        trim=True,
        for_inference=False,
        for_segmentation=False,
        **kwargs,
    ):
        super().__init__()  # not support this kind of **kwargs for now

        self.use_amp = use_amp
        self.for_inference = for_inference
        self.for_segmentation = for_segmentation
        self.trimmer = SequenceTrimmer(enabled=trim and not for_inference)
        self.net = LGATrWrapper(**kwargs)

    def forward(self, x, v=None, mask=None):
        with torch.no_grad():
            x, v, mask, _ = self.trimmer(x, v, mask)

        with torch.autocast("cuda", enabled=self.use_amp):
            output = self.net(x, v, mask)

            if self.for_segmentation:
                output = output.transpose(1, 2).contiguous()
                if self.for_inference:
                    output = torch.softmax(output, dim=1)
                return output

            if self.for_inference:
                output = torch.softmax(output, dim=-1)
            return output
