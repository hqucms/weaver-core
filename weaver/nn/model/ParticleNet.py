''' ParticleNet

Paper: "ParticleNet: Jet Tagging via Particle Clouds" - https://arxiv.org/abs/1902.08570

Adapted from the DGCNN implementation in https://github.com/WangYueFt/dgcnn/blob/master/pytorch/model.py.
'''
import numpy as np
import torch
import torch.nn as nn
from functools import partial

from weaver.utils.logger import _logger
from .ParticleTransformer import delta_phi, SequenceTrimmer


def pairwise_distance(x):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    return (xx.transpose(2, 1) + inner + xx).clamp(min=0)


def pairwise_distance_etaphi(x):
    eta, phi = x.split((1, 1), dim=1)
    dphi = delta_phi(phi.unsqueeze(-1), phi.unsqueeze(-2))
    dist = dphi.square().sum(dim=1) + pairwise_distance(eta)
    return dist


def knn(x, k, distance_fn=pairwise_distance, exclude_self=True):
    # distance: (b, i, j)
    if exclude_self:
        idx = distance_fn(x).topk(k=k + 1, dim=-1, largest=False, sorted=True)[1][:, :, 1:]  # (b, i, k)
    else:
        idx = distance_fn(x).topk(k=k, dim=-1, largest=False, sorted=True)[1]  # (b, i, k)
    return idx


def gather(x, k, idx, cpu_mode=False):
    batch_size, num_dims, num_points = x.size()
    idx_base = torch.arange(0, batch_size, device=x.device).view(-1, 1, 1) * num_points
    idx = (idx + idx_base).view(-1)  # (b*i*k,)
    if cpu_mode:
        # -> (num_dims, batch_size, num_points) -> (num_dims, batch_size*num_points)
        fts = x.transpose(0, 1).reshape(num_dims, -1)
        # neighbors: -> (num_dims, batch_size*num_points*k) -> ...
        fts = fts[:, idx].view(num_dims, batch_size, num_points, k)
        fts = fts.transpose(1, 0).contiguous()  # (batch_size, num_dims, num_points, k)
    else:
        # -> (batch_size, num_points, num_dims) -> (batch_size*num_points, num_dims)
        fts = x.transpose(2, 1).reshape(-1, num_dims)
        # neighbors: -> (batch_size*num_points*k, num_dims) -> ...
        fts = fts[idx, :].view(batch_size, num_points, k, num_dims)
        fts = fts.permute(0, 3, 1, 2).contiguous()  # (batch_size, num_dims, num_points, k)
    return fts


def get_graph_feature(x, k, idx, cpu_mode=False):
    batch_size, num_dims, num_points = x.size()
    fts = gather(x, k, idx, cpu_mode=cpu_mode)
    x = x.view(batch_size, num_dims, num_points, 1).repeat(1, 1, 1, k)
    fts = torch.cat((x, fts - x), dim=1)  # ->(batch_size, 2*num_dims, num_points, k)
    return fts


class EdgeConvBlock(nn.Module):
    r"""EdgeConv layer.
    Introduced in "`Dynamic Graph CNN for Learning on Point Clouds
    <https://arxiv.org/pdf/1801.07829>`__".  Can be described as follows:
    .. math::
       x_i^{(l+1)} = \max_{j \in \mathcal{N}(i)} \mathrm{ReLU}(
       \Theta \cdot (x_j^{(l)} - x_i^{(l)}) + \Phi \cdot x_i^{(l)})
    where :math:`\mathcal{N}(i)` is the neighbor of :math:`i`.
    Parameters
    ----------
    in_feat : int
        Input feature size.
    out_feat : int
        Output feature size.
    batch_norm : bool
        Whether to include batch normalization on messages.
    """

    def __init__(self, k, in_feat, out_feats, batch_norm=True, activation=True, cpu_mode=False):
        super(EdgeConvBlock, self).__init__()
        self.k = k
        self.batch_norm = batch_norm
        self.activation = activation
        self.num_layers = len(out_feats)
        self.get_graph_feature = partial(get_graph_feature, cpu_mode=cpu_mode)

        self.convs = nn.ModuleList()
        for i in range(self.num_layers):
            self.convs.append(
                nn.Conv2d(
                    2 * in_feat if i == 0 else out_feats[i - 1], out_feats[i],
                    kernel_size=1, bias=False if self.batch_norm else True))

        if batch_norm:
            self.bns = nn.ModuleList()
            for i in range(self.num_layers):
                self.bns.append(nn.BatchNorm2d(out_feats[i]))

        if activation:
            self.acts = nn.ModuleList()
            for i in range(self.num_layers):
                self.acts.append(nn.ReLU())

        if in_feat == out_feats[-1]:
            self.sc = None
        else:
            self.sc = nn.Conv1d(in_feat, out_feats[-1], kernel_size=1, bias=False)
            self.sc_bn = nn.BatchNorm1d(out_feats[-1])

        if activation:
            self.sc_act = nn.ReLU()

    def forward(self, points, features):

        topk_indices = knn(points, self.k)
        x = self.get_graph_feature(features, self.k, topk_indices)

        for conv, bn, act in zip(self.convs, self.bns, self.acts):
            x = conv(x)  # (N, C', P, K)
            if bn:
                x = bn(x)
            if act:
                x = act(x)

        fts = x.mean(dim=-1)  # (N, C, P)

        # shortcut
        if self.sc:
            sc = self.sc(features)  # (N, C_out, P)
            sc = self.sc_bn(sc)
        else:
            sc = features

        return self.sc_act(sc + fts)  # (N, C_out, P)


class ParticleNet(nn.Module):

    def __init__(self,
                 input_dims,
                 num_classes,
                 conv_params=[(7, (32, 32, 32)), (7, (64, 64, 64))],
                 fc_params=[(128, 0.1)],
                 use_fusion=True,
                 use_fts_bn=True,
                 use_counts=True,
                 trim=True,
                 for_inference=False,
                 for_segmentation=False,
                 use_amp=False,
                 **kwargs):
        super(ParticleNet, self).__init__(**kwargs)

        _logger.info('ParticleNet init-ed: %s', locals())

        self.trimmer = SequenceTrimmer(enabled=trim and not for_inference)

        self.use_fts_bn = use_fts_bn
        if self.use_fts_bn:
            self.bn_fts = nn.BatchNorm1d(input_dims)

        self.use_counts = use_counts

        self.edge_convs = nn.ModuleList()
        for idx, layer_param in enumerate(conv_params):
            k, channels = layer_param
            in_feat = input_dims if idx == 0 else conv_params[idx - 1][1][-1]
            self.edge_convs.append(EdgeConvBlock(k=k, in_feat=in_feat, out_feats=channels, cpu_mode=for_inference))

        self.use_fusion = use_fusion
        if self.use_fusion:
            in_chn = sum(x[-1] for _, x in conv_params)
            out_chn = np.clip((in_chn // 128) * 128, 128, 1024)
            self.fusion_block = nn.Sequential(
                nn.Conv1d(in_chn, out_chn, kernel_size=1, bias=False),
                nn.BatchNorm1d(out_chn),
                nn.ReLU())

        self.for_segmentation = for_segmentation

        fcs = []
        for idx, layer_param in enumerate(fc_params):
            channels, drop_rate = layer_param
            if idx == 0:
                in_chn = out_chn if self.use_fusion else conv_params[-1][1][-1]
            else:
                in_chn = fc_params[idx - 1][0]
            if self.for_segmentation:
                fcs.append(nn.Sequential(nn.Conv1d(in_chn, channels, kernel_size=1, bias=False),
                                         nn.BatchNorm1d(channels), nn.ReLU(), nn.Dropout(drop_rate)))
            else:
                fcs.append(nn.Sequential(nn.Linear(in_chn, channels), nn.ReLU(), nn.Dropout(drop_rate)))
        if self.for_segmentation:
            fcs.append(nn.Conv1d(fc_params[-1][0], num_classes, kernel_size=1))
        else:
            fcs.append(nn.Linear(fc_params[-1][0], num_classes))
        self.fc = nn.Sequential(*fcs)

        self.for_inference = for_inference
        self.use_amp = use_amp
        self.coord_shift = 99. if use_amp else 1e9

    def forward(self, points, features, mask=None):
        # print('points:\n', points)
        # print('features:\n', features)
        with torch.set_grad_enabled(features.requires_grad):
            points, features, mask, _ = self.trimmer(points, features, mask=mask)
            # padding_mask: False = real particle, True = padded particles
            padding_mask = ~mask

            if self.use_counts:
                counts = mask.float().sum(dim=-1)
                counts = torch.max(counts, torch.ones_like(counts))  # >=1

        with torch.autocast('cuda', enabled=self.use_amp):
            if self.use_fts_bn:
                fts = self.bn_fts(features).masked_fill(padding_mask, 0)
            else:
                fts = features
            outputs = []
            for idx, conv in enumerate(self.edge_convs):
                pts = (points if idx == 0 else fts).masked_fill(padding_mask, self.coord_shift)
                fts = conv(pts, fts).masked_fill(padding_mask, 0)
                if self.use_fusion:
                    outputs.append(fts)
            if self.use_fusion:
                fts = self.fusion_block(torch.cat(outputs, dim=1)).masked_fill(padding_mask, 0)

            # assert(((fts.abs().sum(dim=1, keepdim=True) != 0).float() - mask.float()).abs().sum().item() == 0)

            if self.for_segmentation:
                x = fts
            else:
                if self.use_counts:
                    x = fts.sum(dim=-1) / counts  # divide by the real counts
                else:
                    x = fts.mean(dim=-1)

            output = self.fc(x)
            if self.for_inference:
                output = torch.softmax(output, dim=1)
            # print('output:\n', output)
            return output


class FeatureConv(nn.Module):

    def __init__(self, in_chn, out_chn, **kwargs):
        super(FeatureConv, self).__init__(**kwargs)
        self.conv = nn.Sequential(
            nn.BatchNorm1d(in_chn),
            nn.Conv1d(in_chn, out_chn, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_chn),
            nn.ReLU()
        )

    def forward(self, x):
        return self.conv(x)


class ParticleNetTagger(nn.Module):

    def __init__(self,
                 pf_features_dims,
                 sv_features_dims,
                 num_classes,
                 conv_params=[(7, (32, 32, 32)), (7, (64, 64, 64))],
                 fc_params=[(128, 0.1)],
                 use_fusion=True,
                 use_fts_bn=True,
                 use_counts=True,
                 trim=True,
                 pf_input_dropout=None,
                 sv_input_dropout=None,
                 for_inference=False,
                 use_amp=False,
                 **kwargs):
        super(ParticleNetTagger, self).__init__(**kwargs)
        self.pf_trimmer = SequenceTrimmer(enabled=trim and not for_inference)
        self.sv_trimmer = SequenceTrimmer(enabled=trim and not for_inference)
        self.pf_input_dropout = nn.Dropout(pf_input_dropout) if pf_input_dropout else None
        self.sv_input_dropout = nn.Dropout(sv_input_dropout) if sv_input_dropout else None
        self.pf_conv = FeatureConv(pf_features_dims, 32)
        self.sv_conv = FeatureConv(sv_features_dims, 32)
        self.pn = ParticleNet(input_dims=32,
                              num_classes=num_classes,
                              conv_params=conv_params,
                              fc_params=fc_params,
                              use_fusion=use_fusion,
                              use_fts_bn=use_fts_bn,
                              use_counts=use_counts,
                              trim=False,
                              for_inference=for_inference,
                              use_amp=use_amp)

    def forward(self, pf_points, pf_features, pf_mask, sv_points, sv_features, sv_mask):
        with torch.no_grad():
            pf_points, pf_features, pf_mask, _ = self.pf_trimmer(pf_points, pf_features, pf_mask)
            sv_points, sv_features, sv_mask, _ = self.sv_trimmer(sv_points, sv_features, sv_mask)

        if self.pf_input_dropout:
            pf_mask = (self.pf_input_dropout(pf_mask) != 0).float()
            pf_points *= pf_mask
            pf_features *= pf_mask
        if self.sv_input_dropout:
            sv_mask = (self.sv_input_dropout(sv_mask) != 0).float()
            sv_points *= sv_mask
            sv_features *= sv_mask

        with torch.autocast('cuda', enabled=self.use_amp):
            points = torch.cat((pf_points, sv_points), dim=2)
            features = torch.cat((self.pf_conv(pf_features * pf_mask) * pf_mask,
                                  self.sv_conv(sv_features * sv_mask) * sv_mask), dim=2)
            mask = torch.cat((pf_mask, sv_mask), dim=2)
            return self.pn(points, features, mask)
