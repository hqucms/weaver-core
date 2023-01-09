''' ParticleNeXt

An improved version of ParticleNet (https://arxiv.org/abs/1902.08570).

See talk at ML4Jets2021: https://indico.cern.ch/event/980214/contributions/4413544
'''
import math
import numpy as np
import random
import torch
import torch.nn as nn
from functools import partial


def delta_phi(a, b):
    return (a - b + math.pi) % (2 * math.pi) - math.pi


def delta_r2(eta1, phi1, eta2, phi2):
    return (eta1 - eta2)**2 + delta_phi(phi1, phi2)**2


def to_pt2(x, eps=1e-8):
    pt2 = x[:, :2].square().sum(dim=1, keepdim=True)
    if eps is not None:
        pt2 = pt2.clamp(min=eps)
    return pt2


def to_m2(x, eps=1e-8):
    m2 = x[:, 3:4].square() - x[:, :3].square().sum(dim=1, keepdim=True)
    if eps is not None:
        m2 = m2.clamp(min=eps)
    return m2


def atan2(y, x):
    sx = torch.sign(x)
    sy = torch.sign(y)
    pi_part = (sy + sx * (sy ** 2 - 1)) * (sx - 1) * (-math.pi / 2)
    atan_part = torch.arctan(y / (x + (1 - sx ** 2))) * sx ** 2
    return atan_part + pi_part


def to_ptrapphim(x, return_mass=True, eps=1e-8, for_onnx=False):
    # x: (N, 4, ...), dim1 : (px, py, pz, E)
    px, py, pz, energy = x.split((1, 1, 1, 1), dim=1)
    pt = torch.sqrt(to_pt2(x, eps=eps))
    # rapidity = 0.5 * torch.log((energy + pz) / (energy - pz))
    rapidity = 0.5 * torch.log(1 + (2 * pz) / (energy - pz).clamp(min=1e-20))
    phi = (atan2 if for_onnx else torch.atan2)(py, px)
    if not return_mass:
        return torch.cat((pt, rapidity, phi), dim=1)
    else:
        m = torch.sqrt(to_m2(x, eps=eps))
        return torch.cat((pt, rapidity, phi, m), dim=1)


def to_xyzt(x):
    # x: (N, 4, ...), dim1 : (pt, rap, phi, E)
    pt, rapidity, phi, energy = x.split((1, 1, 1, 1), dim=1)
    px = pt * torch.cos(phi)
    py = pt * torch.sin(phi)
    pz = energy * torch.tanh(rapidity)
    return torch.cat((px, py, pz, energy), dim=1)


def boost(x, boostp4, eps=1e-8):
    # boost x to the rest frame of boostp4
    # x: (N, 4, ...), dim1 : (px, py, pz, E)
    p3 = -boostp4[:, :3] / boostp4[:, 3:].clamp(min=eps)
    b2 = p3.square().sum(dim=1, keepdim=True)
    gamma = (1 - b2).clamp(min=eps)**(-0.5)
    gamma2 = (gamma - 1) / b2
    gamma2.masked_fill_(b2 == 0, 0)
    bp = (x[:, :3] * p3).sum(dim=1, keepdim=True)
    v = x[:, :3] + gamma2 * bp * p3 + x[:, 3:] * gamma * p3
    return v


def p3_norm(p, eps=1e-8):
    return p[:, :3] / p[:, :3].norm(dim=1, keepdim=True).clamp(min=eps)


def pairwise_distance(x):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    return (xx.transpose(2, 1) + inner + xx).clamp(min=0)


def pairwise_distance_etaphi(x):
    eta, phi = x.split((1, 1), dim=1)
    dphi = delta_phi(phi.unsqueeze(-1), phi.unsqueeze(-2))
    dist = dphi.square().sum(dim=1) + pairwise_distance(eta)
    return dist


def knn(x, k, distance_fn=pairwise_distance_etaphi, exclude_self=False):
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


def gather_edges(x, idx):
    num_dims = x.size(1)
    idx = idx.unsqueeze(1).repeat(1, num_dims, 1, 1)
    return x.gather(-1, idx)


def pairwise_lv_fts(xi, xj, use_polarization_angle=False, eps=1e-8, for_onnx=False):
    pti, rapi, phii = to_ptrapphim(xi, False, eps=None, for_onnx=for_onnx).split((1, 1, 1), dim=1)
    ptj, rapj, phij = to_ptrapphim(xj, False, eps=None, for_onnx=for_onnx).split((1, 1, 1), dim=1)

    ptmin = ((pti <= ptj) * pti + (pti > ptj) * ptj) if for_onnx else torch.minimum(pti, ptj)
    delta = delta_r2(rapi, phii, rapj, phij).sqrt()
    lndelta = torch.log(delta.clamp(min=eps))
    lnkt = torch.log((ptmin * delta).clamp(min=eps))
    lnz = torch.log((ptmin / (pti + ptj).clamp(min=eps)).clamp(min=eps))

    xij = xi + xj
    lnm2 = torch.log(to_m2(xij, eps=eps))

    outputs = [lnkt, lnz, lndelta, lnm2]

    if use_polarization_angle:
        xj_boost = boost(xj, xij)
        costheta = (p3_norm(xj_boost, eps=eps) * p3_norm(xij, eps=eps)).sum(dim=1, keepdim=True)
        outputs.append(costheta)

    return torch.cat(outputs, dim=1)


def get_graph_feature(pts=None, fts=None, lvs=None, mask=None, edges=None, idx=None, null_edge_pos=None,
                      k=None,
                      use_rel_fts=False,
                      use_rel_coords=False,
                      use_rel_dist=False,
                      use_rel_lv_fts=True,
                      use_polarization_angle=False,
                      cpu_mode=False, eps=1e-8):
    if null_edge_pos is None and mask is not None:
        mask_ngbs = gather(mask, k, idx, cpu_mode=cpu_mode)
        mask_ngbs = mask_ngbs & mask.unsqueeze(-1)
        null_edge_pos = ~mask_ngbs

    outputs = []
    if fts is not None:
        fts_ngbs = gather(fts, k, idx, cpu_mode=cpu_mode)
        fts_center = fts.unsqueeze(-1).repeat(1, 1, 1, k)
        outputs.extend([fts_center, (fts_ngbs - fts_center) if use_rel_fts else fts_ngbs])

    rel_coords = None
    if pts is not None:
        if use_rel_coords or use_rel_dist:
            pts_ngbs = gather(pts, k, idx, cpu_mode=cpu_mode)
            rel_coords = pts_ngbs - pts.unsqueeze(-1)
            if use_rel_dist:
                rel_dist = torch.norm(rel_coords, dim=1, keepdim=True)
                outputs.append(torch.log(rel_dist.clamp(min=eps)))

    lvs_ngbs = None
    if lvs is not None:
        if use_rel_lv_fts:
            lvs_ngbs = gather(lvs, k, idx, cpu_mode=cpu_mode)
            outputs.append(
                pairwise_lv_fts(lvs.unsqueeze(-1).repeat(1, 1, 1, k), lvs_ngbs,
                                use_polarization_angle=use_polarization_angle,
                                for_onnx=cpu_mode))

    if edges is not None:
        outputs.append(gather_edges(edges, idx))

    if len(outputs) > 0:
        # (batch_size, c, num_points, k)
        outputs = torch.cat(outputs, dim=1)
    else:
        outputs = None

    return outputs, rel_coords, lvs_ngbs, null_edge_pos


# https://github.com/moskomule/senet.pytorch/blob/master/senet/se_module.py
class SqueezeAndExcitation2d(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SqueezeAndExcitation2d, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x, mask):
        with torch.no_grad():
            n = mask.sum((2, 3))
            n.masked_fill_(n == 0, 1)

        b, c, _, _ = x.size()
        y = (x * mask).sum((2, 3)) / n
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class SqueezeAndExcitation1d(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SqueezeAndExcitation1d, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x, mask):
        with torch.no_grad():
            n = mask.sum(2)
            n.masked_fill_(n == 0, 1)

        b, c, _ = x.size()
        y = (x * mask).sum(2) / n
        y = self.fc(y).view(b, c, 1)
        return x * y


class MultiScaleEdgeConv(nn.Module):

    def __init__(self, node_dim, edge_dim,
                 num_neighbors, out_dim,
                 reduction_dilation=None,
                 message_dim=None,
                 # more options
                 edge_aggregation='attn',
                 use_rel_lv_fts=True,
                 use_polarization_angle=False,
                 use_rel_fts=False,
                 use_rel_dist=False,
                 update_coords=False,
                 lv_aggregation=False,
                 use_node_se=True,
                 use_edge_se=True,
                 init_scale=1e-5,
                 cpu_mode=False,
                 ):
        super(MultiScaleEdgeConv, self).__init__()

        self.update_coords = update_coords
        self.lv_aggregation = lv_aggregation

        self.k = num_neighbors
        self.num_neighbors_in = num_neighbors
        self.knn = partial(knn, k=num_neighbors)
        self.get_graph_feature = partial(get_graph_feature, k=num_neighbors,
                                         use_rel_fts=use_rel_fts,
                                         use_rel_coords=update_coords,
                                         use_rel_dist=use_rel_dist,
                                         use_rel_lv_fts=use_rel_lv_fts,
                                         use_polarization_angle=use_polarization_angle,
                                         cpu_mode=cpu_mode)

        self.slices = []
        self.slice_dims = []
        if reduction_dilation is None:
            reduction_dilation = [(1, 1)]
        for reduction, dilation in reduction_dilation:
            self.slices.append(slice(None, num_neighbors // reduction, dilation))
            self.slice_dims.append(num_neighbors // reduction // dilation)

        if message_dim is None:
            message_dim = out_dim

        self.node_encode = nn.Sequential(
            nn.BatchNorm2d(node_dim),
            nn.ReLU(),
            nn.Conv2d(node_dim, message_dim // 2, kernel_size=1, bias=False),
        ) if node_dim != message_dim // 2 else nn.Identity()

        edge_input_dim = message_dim + edge_dim
        self.edge_mlp = nn.Sequential(
            nn.BatchNorm2d(edge_input_dim),
            nn.ReLU(),
            nn.Conv2d(edge_input_dim, message_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(message_dim),
            nn.ReLU(),
            nn.Conv2d(message_dim, message_dim, kernel_size=1, bias=False),
        )
        self.edge_se = SqueezeAndExcitation2d(message_dim) if use_edge_se else None

        self.coords_mlp = nn.ModuleList([nn.Sequential(
            nn.BatchNorm2d(message_dim * len(self.slices)),
            nn.ReLU(),
            nn.Conv2d(message_dim * len(self.slices), 1, kernel_size=1, bias=False),
        ) for _ in self.slices]) if self.update_coords else None

        node_input_dim = message_dim * len(self.slices)

        self.edge_aggregation = edge_aggregation.lower()
        if 'attn' not in self.edge_aggregation and self.edge_aggregation not in ('sum', 'mean', 'max'):
            raise RuntimeError("`edge_aggregation` can only be 'sum', 'mean', 'max', 'attn'")
        if 'attn' in self.edge_aggregation:
            attn_dim = message_dim if self.edge_aggregation == 'attn' else int(self.edge_aggregation[4:])
            self.sa = nn.ModuleList([nn.Sequential(
                nn.BatchNorm2d(message_dim),
                nn.ReLU(),
                nn.Conv2d(message_dim, attn_dim, kernel_size=1, bias=False)
            ) for _ in self.slices])
            self.sa_repeat = None if (attn_dim == 1 or attn_dim >= message_dim) else message_dim // attn_dim
            self.msg_repeat = None if attn_dim <= message_dim else attn_dim // message_dim
            node_input_dim = max(message_dim, attn_dim) * len(self.slices)

        if self.lv_aggregation:
            self.lv_sa = nn.ModuleList([nn.Sequential(
                nn.BatchNorm2d(message_dim),
                nn.ReLU(),
                nn.Conv2d(message_dim, 1, kernel_size=1, bias=False)
            ) for _ in self.slices])

            node_lv_dim = 2 * len(self.slices)
            self.lv_encode = nn.Sequential(
                nn.BatchNorm2d(node_lv_dim),
                nn.ReLU(),
                nn.Conv2d(node_lv_dim, node_lv_dim, kernel_size=1, bias=False),
            )
            node_input_dim += node_lv_dim

        self.node_mlp = nn.Sequential(
            nn.BatchNorm2d(node_input_dim),
            nn.ReLU(),
            nn.Conv2d(node_input_dim, out_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(),
            nn.Conv2d(out_dim, out_dim, kernel_size=1, bias=False),
        )
        self.node_se = SqueezeAndExcitation2d(out_dim) if use_node_se else None

        self.shortcut = nn.Sequential(
            nn.Conv2d(node_dim, out_dim, kernel_size=1, bias=False),
        ) if node_dim != out_dim else nn.Identity()

        self.gamma = nn.Parameter(init_scale * torch.ones((out_dim, 1, 1)))

    def forward(self, points, features, lorentz_vectors, mask=None, edges=None,
                idx=None, null_edge_pos=None, edge_inputs=None, lvs_ngbs=None):

        fts_encode = self.node_encode(features).squeeze(-1)

        if idx is None:
            idx = self.knn(points)
            edges, rel_coords, lvs_ngbs, null_edge_pos = self.get_graph_feature(
                points, fts_encode, lorentz_vectors, mask, edges, idx, null_edge_pos)
        else:
            if self.k < self.num_neighbors_in:
                idx = idx[:, :, :self.k]
                null_edge_pos = null_edge_pos[:, :, :, :self.k]
                if edge_inputs is not None:
                    edge_inputs = edge_inputs[:, :, :, :self.k]
            edges, *_ = self.get_graph_feature(
                fts=fts_encode, mask=mask, edges=edges, idx=idx, null_edge_pos=null_edge_pos)
            if edge_inputs is not None:
                edges = torch.cat([edges, edge_inputs], dim=1)

        if sum(self.slice_dims) < self.k:
            edges = torch.cat([edges[:, :, :, s] for s in self.slices], dim=-1)
            null_edge_pos = torch.cat([null_edge_pos[:, :, :, s] for s in self.slices], dim=-1)

        message = self.edge_mlp(edges)
        if self.edge_se is not None:
            message = self.edge_se(message, ~null_edge_pos)

        if sum(self.slice_dims) < self.k:
            message = message.split(self.slice_dims, dim=-1)
            null_edge_pos = null_edge_pos.split(self.slice_dims, dim=-1)
        else:
            message = [message[:, :, :, s] for s in self.slices]
            null_edge_pos = [null_edge_pos[:, :, :, s] for s in self.slices]

        pts_out = points
        node_inputs = []
        node_lv_inputs = []
        for i, s in enumerate(self.slices):
            def masked(x, val=0.):
                return x.masked_fill(null_edge_pos[i], val)

            msg = message[i]

            if self.update_coords:
                coords_weights = masked(self.coords_mlp[i](msg))  # (b, 1, i, k)
                pts_out = pts_out + (coords_weights * rel_coords).sum(-1)

            if 'attn' in self.edge_aggregation:
                attn = masked(self.sa[i](msg), -1e9)
                attn = attn.softmax(dim=-1)
                if self.sa_repeat is not None:
                    attn = attn.repeat(1, self.sa_repeat, 1, 1)
                if self.msg_repeat is not None:
                    msgs = msg.repeat(1, self.msg_repeat, 1, 1)
                else:
                    msgs = msg
                fts = (attn * msgs).sum(dim=-1, keepdim=True)
            else:
                msg = masked(msg)
                if self.edge_aggregation == 'sum':
                    fts = msg.sum(-1, keepdim=True)
                elif self.edge_aggregation == 'max':
                    fts = msg.max(-1, keepdim=True)[0]
                else:
                    fts = msg.mean(-1, keepdim=True)
            node_inputs.append(fts)

            if self.lv_aggregation:
                attn = masked(self.lv_sa[i](msg), -1e9)
                attn = attn.softmax(dim=-1)
                lvs_agg = (attn * lvs_ngbs[:, :, :, s]).sum(-1, keepdim=True)
                lvs = [to_pt2(lvs_agg).log(), to_m2(lvs_agg).log()]
                node_lv_inputs.extend(lvs)

        if self.lv_aggregation:
            node_inputs.append(self.lv_encode(torch.cat(node_lv_inputs, dim=1)))

        node_fts = self.node_mlp(torch.cat(node_inputs, dim=1))
        if self.node_se is not None:
            node_fts = self.node_se(node_fts, mask.unsqueeze(-1))

        fts_out = self.shortcut(features) + self.gamma * node_fts

        return pts_out, fts_out


class ParticleNeXt(nn.Module):

    def __init__(self,
                 feature_input_dim=None,
                 edge_input_dim=0,
                 num_classes=None,
                 # network configurations
                 node_dim=32,
                 edge_dim=8,
                 use_node_bn=True,
                 use_edge_bn=True,
                 layer_params=[(32, 256, [(8, 1), (4, 1), (2, 1), (1, 1)], 64), (32, 256, [(8, 1), (4, 1), (2, 1), (1, 1)], 64), (32, 256, [(8, 1), (4, 1), (2, 1), (1, 1)], 64)],  # noqa
                 fc_params=[(256, 0.1)],
                 global_aggregation='attn4',
                 # MultiScaleEdgeConv options
                 edge_aggregation='attn8',
                 use_rel_lv_fts=True,
                 use_polarization_angle=False,
                 use_rel_fts=False,
                 use_rel_dist=False,
                 update_coords=False,
                 lv_aggregation=False,
                 use_node_se=True,
                 use_edge_se=True,
                 init_scale=1e-5,
                 # input augmentation
                 input_dropout=None,
                 pt_dropout=None,
                 lorentz_vector_scale=None,
                 lorentz_vector_smear=None,
                 lorentz_vector_shift=None,
                 # misc
                 trim=True,
                 for_inference=False,
                 for_segmentation=False,
                 **kwargs):
        super(ParticleNeXt, self).__init__(**kwargs)

        # input augmentation
        self.input_dropout = nn.Dropout(input_dropout) if input_dropout else None
        self.pt_dropout = pt_dropout
        self.lorentz_vector_scale = lorentz_vector_scale
        self.lorentz_vector_smear = lorentz_vector_smear
        self.lorentz_vector_shift = lorentz_vector_shift

        # network configurations
        if node_dim:
            self.node_encode = nn.Sequential(
                nn.BatchNorm2d(feature_input_dim),
                nn.Conv2d(feature_input_dim, node_dim, 1, bias=False)
            )
            input_dim = node_dim
        else:
            self.node_encode = nn.BatchNorm2d(feature_input_dim) if use_node_bn else nn.Identity()
            input_dim = feature_input_dim

        if use_rel_dist:
            edge_input_dim += 1
        if use_rel_lv_fts:
            edge_input_dim += 5 if use_polarization_angle else 4
        if edge_dim and edge_input_dim:
            self.edge_encode = nn.Sequential(
                nn.BatchNorm2d(edge_input_dim),
                nn.Conv2d(edge_input_dim, edge_dim, 1, bias=False)
            )
        else:
            self.edge_encode = nn.BatchNorm2d(edge_input_dim) if use_edge_bn and edge_input_dim > 0 else nn.Identity()
            edge_dim = edge_input_dim

        self.layers = nn.ModuleList()
        num_neighbors = []
        for param in layer_params:
            if isinstance(param, dict):
                k = param['k']
                rd = param['rd']
                out_dim = param['c']
                msg_dim = param.get('m', None)
            elif isinstance(param, (list, tuple)):
                msg_dim = None
                rd = None
                if len(param) == 2:
                    k, out_dim = param
                elif len(param) == 3:
                    k, out_dim, rd = param
                elif len(param) == 4:
                    k, out_dim, rd, msg_dim = param
            self.layers.append(
                MultiScaleEdgeConv(
                    node_dim=input_dim, edge_dim=edge_dim,
                    num_neighbors=k, out_dim=out_dim,
                    reduction_dilation=rd,
                    message_dim=msg_dim,
                    # more options
                    edge_aggregation=edge_aggregation,
                    use_rel_lv_fts=use_rel_lv_fts,
                    use_polarization_angle=use_polarization_angle,
                    use_rel_fts=use_rel_fts,
                    use_rel_dist=use_rel_dist,
                    update_coords=update_coords,
                    lv_aggregation=lv_aggregation,
                    use_node_se=use_node_se,
                    use_edge_se=use_edge_se,
                    init_scale=init_scale,
                    cpu_mode=for_inference,
                )
            )
            num_neighbors.append(k)
            input_dim = out_dim

        self.num_neighbors = max(num_neighbors)
        if not update_coords:
            self.knn = partial(knn, k=self.num_neighbors)
            self.get_graph_feature = partial(get_graph_feature, k=self.num_neighbors,
                                             use_rel_fts=use_rel_fts,
                                             use_rel_coords=update_coords,
                                             use_rel_dist=use_rel_dist,
                                             use_rel_lv_fts=use_rel_lv_fts,
                                             use_polarization_angle=use_polarization_angle,
                                             cpu_mode=for_inference)
            for i in range(len(self.layers)):
                self.layers[i].num_neighbors_in = self.num_neighbors

        else:
            self.knn = None
            self.get_graph_feature = None

        self.post = nn.Sequential(nn.BatchNorm2d(input_dim), nn.ReLU())

        self.global_aggregation = global_aggregation.lower()
        if 'attn' not in self.global_aggregation and self.global_aggregation not in ('mean', 'sum', 'max'):
            raise RuntimeError("`global_aggregation` can only be 'mean', 'sum', 'max', 'attn'")
        if 'attn' in self.global_aggregation:
            attn_dim = input_dim if self.global_aggregation == 'attn' else int(self.global_aggregation[4:])
            self.sa = nn.Conv1d(input_dim, attn_dim, kernel_size=1, bias=False)
            self.sa_repeat = None if (attn_dim == 1 or attn_dim == input_dim) else input_dim // attn_dim

        self.for_segmentation = for_segmentation

        fcs = []
        for out_dim, drop_rate in fc_params:
            if self.for_segmentation:
                fcs.append(
                    nn.Sequential(
                        nn.Conv1d(input_dim, out_dim, kernel_size=1, bias=False),
                        nn.BatchNorm1d(out_dim),
                        nn.ReLU(),
                        nn.Dropout(drop_rate)
                    )
                )
            else:
                fcs.append(nn.Sequential(nn.Linear(input_dim, out_dim), nn.ReLU(), nn.Dropout(drop_rate)))
            input_dim = out_dim
        if self.for_segmentation:
            fcs.append(nn.Conv1d(input_dim, num_classes, kernel_size=1))
        else:
            fcs.append(nn.Linear(input_dim, num_classes))
        self.fc = nn.Sequential(*fcs)

        self.trim = trim
        self.for_inference = for_inference
        self._counter = 0

    def forward(self, points, features, lorentz_vectors, mask=None, edges=None):
        # print('points:\n', points)
        # print('features:\n', features)
        # print('lorentz_vectors:\n', lorentz_vectors)
        # print('mask:\n', mask)

        with torch.no_grad():
            if mask is None:
                mask = torch.ones_like(points[:, 0])

            if self.training:
                if self.input_dropout:
                    mask = self.input_dropout(mask)

                mask = mask.bool()
                b, _, p = mask.size()
                device = mask.device
                if self.pt_dropout:
                    # discard low pt particles, with a random threshold per jet
                    cut = torch.normal(mean=0, std=self.pt_dropout * torch.ones((b, 1, 1), device=device))
                    pt = torch.norm(lorentz_vectors[:, :2], dim=1, keepdim=True)
                    mask &= (pt > cut)

                if self.lorentz_vector_scale:
                    # scale all lorentz vectors in a jet in the same way for each jet
                    scale = torch.normal(mean=1, std=self.lorentz_vector_scale * torch.ones((b, 1, 1), device=device))
                    lorentz_vectors *= scale

                if self.lorentz_vector_smear:
                    # scale lorentz vector randomly for each particle
                    smear = torch.normal(mean=1, std=self.lorentz_vector_smear * torch.ones((b, 1, p), device=device))
                    lorentz_vectors *= smear

                if self.lorentz_vector_shift:
                    # smearing lorentz vector direction randomly for each particle
                    shift = torch.normal(mean=0, std=self.lorentz_vector_shift * torch.ones((b, 2, p), device=device))
                    ptrapphi = to_ptrapphim(lorentz_vectors, False, eps=0)
                    ptrapphi[:, 1:3] += shift
                    lorentz_vectors = to_xyzt(torch.cat((ptrapphi, lorentz_vectors[:, 3:4]), dim=1))

                if self.lorentz_vector_scale is not None or self.lorentz_vector_smear is not None or self.lorentz_vector_shift is not None:
                    # update points and features
                    jet = lorentz_vectors.sum(dim=-1, keepdim=True)
                    jet_rap, jet_phi = to_ptrapphim(jet, False, eps=0)[:, 1:3].split((1, 1), dim=1)
                    pt, rap, phi = to_ptrapphim(lorentz_vectors, False, eps=0).split((1, 1, 1), dim=1)
                    energy = lorentz_vectors[:, 3:4]
                    rap -= jet_rap
                    phi = delta_phi(phi, jet_phi)
                    points = torch.cat((rap, phi), dim=1)
                    features = torch.cat((pt.log(), energy.log(), points), dim=1)

            mask = mask.bool()

            if self.trim and not self.for_inference:
                if self._counter < 5:
                    self._counter += 1
                else:
                    if self.training:
                        q = min(1, random.uniform(0.9, 1.02))
                        maxlen = torch.quantile(mask.float().sum(dim=-1), q).long()
                        rand = torch.rand_like(mask.float())
                        rand.masked_fill_(~mask, -1)
                        perm = rand.argsort(dim=-1, descending=True)
                        mask = torch.gather(mask, -1, perm)
                        points = torch.gather(points, -1, perm.expand_as(points))
                        features = torch.gather(features, -1, perm.expand_as(features))
                        if lorentz_vectors is not None:
                            lorentz_vectors = torch.gather(lorentz_vectors, -1, perm.expand_as(lorentz_vectors))
                        if edges is not None:
                            raise NotImplementedError
                    else:
                        maxlen = mask.sum(dim=-1).max()
                    maxlen = max(maxlen, self.num_neighbors)
                    if maxlen < mask.size(-1):
                        mask = mask[:, :, :maxlen]
                        points = points[:, :, :maxlen]
                        features = features[:, :, :maxlen]
                        if lorentz_vectors is not None:
                            lorentz_vectors = lorentz_vectors[:, :, :maxlen]
                        if edges is not None:
                            edges = edges[:, :, :maxlen, :maxlen]

            if self.global_aggregation == 'mean':
                counts = mask.float().sum(dim=-1)
                counts.masked_fill_(counts == 0, 1)

            null_pos = ~mask

            # push padded points far away
            points.masked_fill_(null_pos, 1e9)
            if self.training:
                # add a random shift to the padded points
                rand_shift = torch.rand_like(points)
                rand_shift.masked_fill_(mask, 0)
                points += 1e6 * rand_shift

            # expand dim of features to use Conv2d (a bit faster than Conv1d)
            features = features.unsqueeze(-1)

            # kNN indices and edge inputs
            if self.knn is not None:
                # using static graph
                idx = self.knn(points)
                edge_inputs, _, lvs_ngbs, null_edge_pos = self.get_graph_feature(
                    lvs=lorentz_vectors, mask=mask, edges=edges, idx=idx, null_edge_pos=None)
                edges = None
            else:
                idx = None
                edge_inputs = None
                lvs_ngbs = None
                null_edge_pos = None

        # === end of no_grad ===

        def masked(x, value=0.):
            return x.masked_fill(null_pos, value)

        # encode features
        features = self.node_encode(features)

        # encode edges
        if edge_inputs is not None:
            edge_inputs = self.edge_encode(edge_inputs)
        elif edges is not None:
            edges = self.edge_encode(edges)

        for layer in self.layers:
            points, features = layer(
                points=points, features=features, lorentz_vectors=lorentz_vectors, mask=mask, edges=edges,
                idx=idx, null_edge_pos=null_edge_pos, edge_inputs=edge_inputs, lvs_ngbs=lvs_ngbs)

        features = self.post(features).squeeze(-1)

        if self.for_segmentation:
            x = masked(features)
        else:
            if 'attn' in self.global_aggregation:
                attn = masked(self.sa(features), -np.inf)
                attn = attn.softmax(dim=-1)
                if self.sa_repeat is not None:
                    attn = attn.repeat(1, self.sa_repeat, 1)
                # assert(((attn.abs().sum(dim=1, keepdim=True) != 0).float() - mask.float()).abs().sum().item() == 0)
                x = (attn * features).sum(dim=-1)
            else:
                features = masked(features)
                # assert(((features.abs().sum(dim=1, keepdim=True) != 0).float() - mask.float()).abs().sum().item() == 0)
                if self.global_aggregation == 'sum':
                    x = features.sum(dim=-1)
                elif self.global_aggregation == 'max':
                    x = features.max(dim=-1)[0]
                else:
                    x = features.sum(dim=-1) / counts

        output = self.fc(x)
        if self.for_inference:
            output = torch.softmax(output, dim=1)
        # print('output:\n', output)
        return output


class ParticleNeXtTagger(nn.Module):

    def __init__(self,
                 pf_features_dims=None,
                 sv_features_dims=None,
                 edge_input_dim=0,
                 num_classes=None,
                 # network configurations
                 node_dim=32,
                 edge_dim=8,
                 use_edge_bn=True,
                 layer_params=[(32, 256, [(8, 1), (4, 1), (2, 1), (1, 1)], 64), (32, 256, [(8, 1), (4, 1), (2, 1), (1, 1)], 64), (32, 256, [(8, 1), (4, 1), (2, 1), (1, 1)], 64)],  # noqa
                 fc_params=[(256, 0.1)],
                 global_aggregation='attn4',
                 # MultiScaleEdgeConv options
                 edge_aggregation='attn8',
                 use_rel_lv_fts=True,
                 use_polarization_angle=False,
                 use_rel_fts=False,
                 use_rel_dist=False,
                 update_coords=False,
                 lv_aggregation=False,
                 use_node_se=True,
                 use_edge_se=True,
                 init_scale=1e-5,
                 # input augmentation
                 pf_input_dropout=None,
                 sv_input_dropout=None,
                 # misc
                 trim=True,
                 for_inference=False,
                 **kwargs):
        super(ParticleNeXtTagger, self).__init__(**kwargs)
        self.pf_input_dropout = nn.Dropout(pf_input_dropout) if pf_input_dropout else None
        self.sv_input_dropout = nn.Dropout(sv_input_dropout) if sv_input_dropout else None
        self.pf_encode = nn.Sequential(
            nn.BatchNorm1d(pf_features_dims),
            nn.Conv1d(pf_features_dims, node_dim, 1, bias=False)
        )
        self.sv_encode = nn.Sequential(
            nn.BatchNorm1d(sv_features_dims),
            nn.Conv1d(sv_features_dims, node_dim, 1, bias=False)
        )
        self.pn = ParticleNeXt(feature_input_dim=node_dim,
                               edge_input_dim=edge_input_dim,
                               num_classes=num_classes,
                               # network configurations
                               node_dim=0,
                               edge_dim=edge_dim,
                               use_node_bn=False,
                               use_edge_bn=use_edge_bn,
                               layer_params=layer_params,
                               fc_params=fc_params,
                               global_aggregation=global_aggregation,
                               # MultiScaleEdgeConv options
                               edge_aggregation=edge_aggregation,
                               use_rel_lv_fts=use_rel_lv_fts,
                               use_polarization_angle=use_polarization_angle,
                               use_rel_fts=use_rel_fts,
                               use_rel_dist=use_rel_dist,
                               update_coords=update_coords,
                               lv_aggregation=lv_aggregation,
                               use_node_se=use_node_se,
                               use_edge_se=use_edge_se,
                               init_scale=init_scale,
                               # misc
                               trim=trim,
                               for_inference=for_inference,
                               )

    def forward(self, pf_points, pf_features, pf_vectors, pf_mask, sv_points, sv_features, sv_vectors, sv_mask, edges=None):
        if self.pf_input_dropout:
            pf_mask = self.pf_input_dropout(pf_mask)
        if self.sv_input_dropout:
            sv_mask = self.sv_input_dropout(sv_mask)

        points = torch.cat((pf_points, sv_points), dim=2)
        features = torch.cat((self.pf_encode(pf_features), self.sv_encode(sv_features)), dim=2)
        lorentz_vectors = torch.cat((pf_vectors, sv_vectors), dim=2)
        mask = torch.cat((pf_mask, sv_mask), dim=2)
        return self.pn(points, features, lorentz_vectors, mask, edges=edges)
