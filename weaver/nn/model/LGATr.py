import torch
from torch import nn

from lgatr import (
    LGATr,
    embed_vector,
    extract_scalar,
    get_num_spurions,
    get_spurions,
    gatr_config,
)
from .ParticleTransformer import SequenceTrimmer


class LGATrWrapper(nn.Module):
    """Interface to the LGATr class"""

    def __init__(
        self,
        in_s_channels: int,
        hidden_mv_channels: int,
        hidden_s_channels: int,
        num_classes: int,
        num_blocks: int,
        num_heads: int,
        # symmetry-breaking configurations
        spurion_token: bool = True,
        beam_spurion: str = "xyplane",
        add_time_spurion: bool = True,
        beam_mirror: bool = True,
        # network configurations
        global_token: bool = True,
        activation: str = "gelu",
        multi_query: bool = False,
        increase_hidden_channels: int = 2,
        head_scale: bool = False,
        double_layernorm: bool = False,
        dropout_prob: float = None,
        # time/memory configurations
        checkpoint_blocks: bool = False,
        # gatr configurations
        use_fully_connected_subgroup: bool = True,
        mix_pseudoscalar_into_scalar: bool = True,
        use_bivector: bool = True,
        use_geometric_product: bool = True,
    ):
        """
        Parameters
        ----------
        in_s_channels : int
            Number of scalar input channels.
            Examples are PID, trajectory displacements and kinematic features
            like log(pT), delta_phi etc that are invariant under z-rotations.
        hidden_mv_channels : int
            Number of hidden multivector channels, defines width of L-GATr.
        hidden_s_channels : int
            Number of hidden scalar channels. We find best performance with
            roughly hidden_s_channels ~ 2 * hidden_mv_channels.
        num_classes : int
            Number of classification scores to predict
        num_blocks : int
            Number of L-GATr blocks.
        num_heads : int
            Number of attention heads in L-GATr.
        spurion_token : bool
            If True, prepend spurions as extra particles (tokens) in the list.
            If False, append spurions as extra mv channels.
        beam_spurion : str
            How the beam spurion is embedded, see lgatr/interface/spurions.py
        add_time_spurion : bool
            If True, add a time spurion.
        beam_mirror : bool
            If True and beam_spurion in ["timelike", "lightlike", "spacelike"],
            add a mirrored beam_spurion, i.e. with opposite p_z.
        global_token : bool
            If True, prepend a global token as first particle in the list.
            If False, fallback to mean-aggregation.
        activation : {"relu", "sigmoid", "gelu"}
            Activation function in the MLP layers.
        multi_query : bool
            If True, use the same query for each head in attention.
        increase_hidden_channels : int
            Factor by which hidden_mv_channels is increased in attention.
        head_scale : bool
            If True, scale the attention heads with a learnable factor.
            Inspired by the NormFormer (https://arxiv.org/pdf/2110.09456)
        double_layernorm : bool
            If True, applies layer normalization also after attention.
            The default is only before attention ('pre-layernorm transformer')
        dropout_prob : float
            Residual dropout after attention and MLP.
        checkpoint_blocks : bool
            If True, use torch.utils.checkpoint.checkpoint to save memory
            at the cost of a slower backward pass.
        use_fully_connected_subgroup : bool
            If True, model is only equivariant with respect to
            the fully connected subgroup of the Lorentz group,
            the proper orthochronous Lorentz group SO^+(1,3),
            which does not include parity and time reversal.
            This setting affects how the EquiLinear maps work:
            For SO^+(1,3), they include transitions scalars/pseudoscalars
            vectors/axialvectors and among bivectors, effectively
            treating the pseudoscalar/axialvector representations
            like another scalar/vector.
            Defaults to False, because parity-odd representations
            are usually not important in high-energy physics simulations.
        mix_pseudoscalar_into_scalar : bool
            If True, the pseudoscalar part of the multivector mixes
            with the pure-scalar channels in the EquiLinear layer.
            This is a technical aspect of how EquiLinear maps work,
            and only makes sense it use_fully_connected_subgroup=True.
            Attention: The combination use_fully_connected_subgroup=False
            and mix_pseudoscalar_into_scalar=True does not make sense,
            you are only equivariant w.r.t. the fully connected subgroup
            if you choose these settings.
        use_bivector : bool
            If False, the bivector components are set to zero after they
            are created in the GeometricBilinear layer.
            This is a toy switch to explore the effect of higher-order
            representations.
        use_geometric_product : bool
            If False, the GeometricBilinear layer is replaced
            by a EquiLinear + ScalarGatedNonlinearity layer.
            This is a toy switch to explore the effect of the geometric product.
        """
        super().__init__()

        # spurion business
        in_mv_channels = 1
        self.global_token = global_token
        self.spurion_token = spurion_token

        num_spurions = get_num_spurions(
            beam_spurion, add_time_spurion, beam_mirror=beam_mirror
        )
        if not self.spurion_token:
            in_mv_channels += num_spurions
        self.spurion_kwargs = {
            "beam_spurion": beam_spurion,
            "add_time_spurion": add_time_spurion,
            "beam_mirror": beam_mirror,
        }

        gatr_config.use_fully_connected_subgroup = use_fully_connected_subgroup
        gatr_config.mix_pseudoscalar_into_scalar = mix_pseudoscalar_into_scalar
        gatr_config.use_bivector = use_bivector
        gatr_config.use_geometric_product = use_geometric_product

        attention = dict(
            multi_query=multi_query,
            num_heads=num_heads,
            increase_hidden_channels=increase_hidden_channels,
            dropout_prob=dropout_prob,
            head_scale=head_scale,
        )
        mlp = dict(
            activation=activation,
            dropout_prob=dropout_prob,
        )

        self.net = LGATr(
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
        """
        Parameters
        ----------
        x : torch.Tensor with shape (batch_size, num_fts, seq_len)
            Scalar features, i.e. features that are invariant under z-rotations.
            Examples: PID, trajectory displacements, kinematic features like
            log(pT), delta_phi, delta_eta
        v : torch.Tensor with shape (batch_size, 4, seq_len)
            Lorentz vectors in format (px, py, pz, E)
        mask : torch.Tensor with shape (batch_size, 1, seq_len)
            Boolean mask that contains 'False' for padded jet constituents

        Returns
        -------
        output : torch.Tensor with shape (batch_size, num_classes)
            Tagging scores for each class
        """
        # reshape input
        x = x.transpose(1, 2)  # (batch_size, seq_len, num_fts)
        v = v.transpose(1, 2)  # (batch_size, seq_len, 4)
        mask = mask.transpose(1, 2)  # (batchsize, seq_len, 1)

        # embed data into geometric algebra
        fourmomenta = v[:, :, None, [3, 0, 1, 2]]  # (px, py, pz, E) -> (E, px, py, pz)
        mv = embed_vector(fourmomenta)  # (batch_size, seq_len, 1, 16)
        s = x  # (batch_size, seq_len, num_fts)

        # symmetry breaking with spurions
        spurions = get_spurions(**self.spurion_kwargs).to(
            device=s.device, dtype=s.dtype
        )
        if self.spurion_token:
            # prepend spurions as extra particles in the list
            mask_ones = torch.ones_like(mask[:, [0]]).repeat(1, spurions.shape[0], 1)
            mask = torch.cat([mask_ones, mask], dim=1)
            s_zeros = torch.zeros_like(s[:, [0]]).repeat(1, spurions.shape[0], 1)
            s = torch.cat([s_zeros, s], dim=1)
            spurions = spurions[None, :, None, :].repeat(mv.shape[0], 1, 1, 1)
            mv = torch.cat([spurions, mv], dim=1)
        else:
            # append spurions as extra mv channels
            spurions = spurions[None, None, :, :].repeat(mv.shape[0], mv.shape[1], 1, 1)
            mv = torch.cat([mv, spurions], dim=2)

        if self.global_token:
            # prepend global token as first particle in the list
            global_token = torch.zeros_like(mv[:, [0], :, :])
            mv = torch.cat((global_token, mv), dim=1)
            mask_ones = torch.ones_like(mask[:, [0]])
            mask = torch.cat((mask_ones, mask), dim=1)
            s_zeros = torch.zeros_like(s[:, [0]])
            s = torch.cat((s_zeros, s), dim=1)
            is_global = torch.zeros_like(s[:, :, 0], dtype=torch.bool)
            is_global[:, 0] = True

        # reshape mask to broadcast correctly
        mask = mask.bool()
        attn_mask = mask[:, None, None, :, 0]  # (batch_size, 1, 1, seq_len)
        attn_kwargs = {"attn_mask": attn_mask}

        # call network
        out_mv, _ = self.net(mv, s, **attn_kwargs)
        output = extract_scalar(out_mv)[..., 0]  # (batch_size, seq_len, num_classes)

        # aggregation
        if self.global_token:
            output = output[is_global]
        else:
            # mean aggregation
            output[~mask[:, 0, 0]] = 0.0
            output = output.mean(dim=1)
        return output


class LGATrTagger(nn.Module):
    """Mimic other weaver wrappers"""

    def __init__(
        self,
        use_amp=False,
        trim=True,
        for_inference=False,
        for_segmentation=False,
        **kwargs,
    ):
        super().__init__()

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
