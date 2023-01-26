import numpy as np
import torch
import torch.nn as nn

import sys

'''orig_stdout = sys.stdout
f = open('out_new.txt', 'w')
sys.stdout = f'''

torch.set_printoptions(profile="full")
#np.set_printoptions(threshold=sys.maxsize)
#torch.set_printoptions(edgeitems=5)

NEW_EDGES=False
NEW_EDGES_DUMMY=False
EDGE_FEATURES=False
EDGE_FEATURES_OR_CONNECTION=True
EDGE_FEATURES_OR_CONNECTION_DUMMY=False

'''Based on https://github.com/WangYueFt/dgcnn/blob/master/pytorch/model.py.'''


def build_sparse_tensor(uu, idx, seq_len):
    # inputs: uu (N, C, num_pairs), idx (N, 2, num_pairs)
    # return: (N, C, seq_len, seq_len) (seq_len is the number of pf+sv)
    batch_size, num_fts, num_pairs = uu.size()

    #print("edgeList build_sparse_tensor\n", idx.size() , "\n", idx)
    #print("edgefts build_sparse_tensor\n", uu.size() , "\n", uu)

    idx = torch.min(idx, torch.ones_like(idx) * seq_len)
    i = torch.cat((
        torch.arange(0, batch_size, device=uu.device).repeat_interleave(num_fts * num_pairs).unsqueeze(0),
        torch.arange(0, num_fts, device=uu.device).repeat_interleave(num_pairs).repeat(batch_size).unsqueeze(0),
        idx[:, :1, :].expand_as(uu).flatten().unsqueeze(0),
        idx[:, 1:, :].expand_as(uu).flatten().unsqueeze(0),
    ), dim=0)

    ef_sparse_tensor = torch.sparse_coo_tensor(
        i, uu.flatten(),
        size=(batch_size, num_fts, seq_len + 1, seq_len + 1),
        device=uu.device).to_dense()[:, :, :seq_len, :seq_len]

    #print("ef_sparse_tensor build_sparse_tensor\n", ef_sparse_tensor.size() , "\n", ef_sparse_tensor)

    return ef_sparse_tensor


def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    idx = pairwise_distance.topk(k=k + 1, dim=-1)[1][:, :, 1:]  # (batch_size, num_points, k)
    return idx


def get_edges(el, bs, np, k_ef):

    #print("edgeList get_edges\n", el.size() , "\n", el)
    idx_tensor= torch.arange(np-k_ef, np, device=el.device).repeat(bs, np, 1)
    for batch in range(bs):
        j=0
        for i, pf in enumerate(el[batch, 0]):
            if i!=0 and pf_idx != int(pf.item()):
                if int(pf.item())==0:
                    break
                j=0
            if j>=k_ef:
                continue
            pf_idx = int(pf.item())
            idx_tensor[batch, pf_idx, j] = el[batch, 1, i]
            j+=1

    return idx_tensor

"""
def prova():
    edge_list = edge_features[:, :2, :]#.type(torch.LongTensor) # track1_index and track2_index
    #print("edgeList get_edges\n", edge_list.size() , "\n", edge_list)
    batch_size, num_dims, num_points = features.size()
    idx_tensor= torch.arange(num_points-k_ef, num_points, device=edge_list.device).repeat(batch_size, num_points, 1)
    #j=0
    j = torch.arange(0, k_ef, device=edge_list.device).repeat(batch_size, num_points, 1)
    #print(edge_list[:, 0, :])
    for i, pf in enumerate(edge_list[:, 0, :]):
        #print(pf)
        if i!=0 and not torch.eq(pf_idx ,pf):
            if int(pf.item())==0:
                j=0
                break
            j=0
        if j>=k_ef:
            j=0
            continue
        pf_idx=pf[:].type(torch.int64)
        #pf_idx = pf.int()
        idx_tensor[:, pf_idx[:][i], j] = edge_list[:, 1, i]
        j+=1


    edge_list = edge_features[:, :2, :].type(torch.int64) # track1_index and track2_index
    #print("edgeList get_edges\n", edge_list.size() , "\n", edge_list)
    batch_size, num_dims, num_points = features.size()
    idx_tensor= torch.arange(num_points-k_ef, num_points, device=edge_list.device).repeat(batch_size, num_points, 1)
    #print("idx_tensor get_edges\n", idx_tensor.size() , "\n", idx_tensor)


    #print(torch.unsqueeze(edge_list[:, 0, :], dim=1).size())
    idx_tensor.scatter_(1, torch.unsqueeze(edge_list[:, 0, :], dim=1), torch.unsqueeze(edge_list[:, 1, :], dim=1))

    #idx2_tensor=edge_list[:, 1, :].repeat(1, num_points, 1)

    #idx_tensor[edge_list[:, 0, :]] = idx2_tensor[edge_list[:, 0, :]]

    #print("idx_tensor new get_edges\n", idx_tensor.size() , "\n", idx_tensor)"""

def edges_or(idx_tensor, el, bs, np, k_ef, efs):
    #print("edgeList get_edges\n", el.size() , "\n", el)

    #print("idx get_edges\n", idx.size() , "\n", idx)
    #print("efs \n", efs.size() , "\n", efs)

    efs = efs.transpose(2, 1) #(batch_size, lenght_edgefeatures, num_ef)
    #print("efs new\n", efs.size() , "\n", efs)

    num_ef=efs.size()[2]
    efs_tensor= torch.zeros(bs, num_ef, np, k_ef, device=efs.device)

    for batch in range(bs):
        j=0
        for i, pf in enumerate(el[batch, 0]):
            if i!=0 and pf_idx != int(pf.item()):
                if int(pf.item())==0:
                    break
                j=0
            if j>=k_ef:
                continue
            pf_idx = int(pf.item())
            idx_tensor[batch, pf_idx, j] = el[batch, 1, i]
            for h, ef in enumerate(efs[batch, i]):
                efs_tensor[batch, h, pf_idx, j] = ef

            j+=1




    #print("efs_tensor \n", efs_tensor.size() , "\n", efs_tensor)
    #print("idx new get_edges\n", idx.size() , "\n", idx)
    return idx_tensor, efs_tensor


def edges_or_new_dummy(idx_tensor, el, bs, np, k_ef, efs):
    #print("edgeList get_edges\n", el.size() , "\n", el)

    #print("idx_tensor get_edges\n", idx_tensor.size() , "\n", idx_tensor)
    #print("efs \n", efs.size() , "\n", efs)

    efs = efs.transpose(2, 1) #(batch_size, lenght_edgefeatures, num_ef)
    #print("efs new\n", efs.size() , "\n", efs)

    num_ef=efs.size()[2]
    efs_tensor= torch.zeros(bs, num_ef, np, k_ef, device=efs.device)
    idx_tensor_final= torch.arange(np-k_ef, np, device=el.device).repeat(bs, np, 1)

    for batch in range(bs):
        j=0
        for i, pf in enumerate(el[batch, 0]):
            if i!=0 and pf_idx != int(pf.item()):
                if int(pf.item())==0:
                    break
                j=0
            if j>=k_ef:
                continue
            pf_idx = int(pf.item())
            idx_tensor_final[batch, pf_idx, j] = el[batch, 1, i]
            for h, ef in enumerate(efs[batch, i]):
                efs_tensor[batch, h, pf_idx, j] = ef

            j+=1
        for pf_idx in range(np):
            if idx_tensor_final[batch, pf_idx, 0] == np-k_ef:
                idx_tensor_final[batch, pf_idx] = idx_tensor[batch, pf_idx]

    #print("efs_tensor \n", efs_tensor.size() , "\n", efs_tensor)
    #print("idx_tensor_final get_edges\n", idx_tensor_final.size() , "\n", idx_tensor_final)
    return idx_tensor_final, efs_tensor


def edges_or_new(idx_tensor, el, bs, np, k_ef, efs):
    #print("edgeList get_edges\n", el.size() , "\n", el)

    #print("idx_tensor get_edges\n", idx_tensor.size() , "\n", idx_tensor)
    #print("efs \n", efs.size() , "\n", efs)

    efs = efs.transpose(2, 1) #(batch_size, lenght_edgefeatures, num_ef)
    #print("efs new\n", efs.size() , "\n", efs)

    num_ef=efs.size()[2]
    efs_tensor= torch.zeros(bs, num_ef, np, k_ef, device=efs.device)
    idx_tensor_final= torch.zeros(bs, np, k_ef, device=efs.device).type(torch.int64)+np-1

    for batch in range(bs):
        j=0
        for i, pf in enumerate(el[batch, 0]):
            if i!=0 and pf_idx != int(pf.item()):
                if int(pf.item())==0:
                    break
                j=0
            if j>=k_ef:
                continue
            pf_idx = int(pf.item())
            idx_tensor_final[batch, pf_idx, j] = el[batch, 1, i]
            for h, ef in enumerate(efs[batch, i]):
                efs_tensor[batch, h, pf_idx, j] = ef

            j+=1
        for pf_idx in range(np):
            if idx_tensor_final[batch, pf_idx, 0] == np-1:
                idx_tensor_final[batch, pf_idx] = idx_tensor[batch, pf_idx]

    #print("efs_tensor \n", efs_tensor.size() , "\n", efs_tensor)
    #print("idx_tensor_final get_edges\n", idx_tensor_final.size() , "\n", idx_tensor_final)
    return idx_tensor_final, efs_tensor


# v1 is faster on GPU
def get_graph_feature_v1(x, k, idx):
    batch_size, num_dims, num_points = x.size()
    #print("x get graph\n ", x.size(),  "\n"  , x)

    #print("idx get graph\n ", idx.size(),  "\n"  , idx)  # (batch_size, num_points, k)
    idx_base = torch.arange(0, batch_size, device=x.device).view(-1, 1, 1) * num_points # (batch_size, 1, 1)
    #print("idx_base get graph\n ", idx_base.size(),  "\n"  , idx_base)
    idx = idx + idx_base # for each batch increases the index of a quantity equal to num_points
    #print("idx new get graph\n ", idx.size(),  "\n"  , idx)
    idx = idx.view(-1) # (batch_size*num_points*k)
    #print("idx new new get graph\n ", idx.size(),  "\n"  , idx)


    fts = x.transpose(2, 1).reshape(-1, num_dims)  # (batch_size, num_dims, num_points)-> (batch_size, num_points, num_dims) -> (batch_size*num_points, num_dims)
    #print("fts  get graph\n ", fts.size(),  "\n"  , fts)
    fts = fts[idx, :].view(batch_size, num_points, k, num_dims)  # neighbors: -> (batch_size*num_points*k, num_dims) -> ...
    #print("fts new get graph\n ", fts.size(),  "\n"  , fts)
    fts = fts.permute(0, 3, 1, 2).contiguous()  # (batch_size, num_dims, num_points, k)
    #print("fts new new get graph\n ", fts.size(),  "\n"  , fts)
    x = x.view(batch_size, num_dims, num_points, 1).repeat(1, 1, 1, k)
    #print("x new get graph\n ", x.size(),  "\n"  , x)

    fts = torch.cat((x, fts - x), dim=1)  # ->(batch_size, 2*num_dims, num_points, k)
    #print("fts last get graph\n ", fts.size(),  "\n"  , fts)

    return fts


# v2 is faster on CPU
def get_graph_feature_v2(x, k, idx):
    batch_size, num_dims, num_points = x.size()

    idx_base = torch.arange(0, batch_size, device=x.device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)

    fts = x.transpose(0, 1).reshape(num_dims, -1)  # -> (num_dims, batch_size, num_points) -> (num_dims, batch_size*num_points)
    fts = fts[:, idx].view(num_dims, batch_size, num_points, k)  # neighbors: -> (num_dims, batch_size*num_points*k) -> ...
    fts = fts.transpose(1, 0).contiguous()  # (batch_size, num_dims, num_points, k)

    x = x.view(batch_size, num_dims, num_points, 1).repeat(1, 1, 1, k)
    fts = torch.cat((x, fts - x), dim=1)  # ->(batch_size, 2*num_dims, num_points, k)

    return fts

"""
for j in range((edgeList.size())[0]):
        G = networkx.DiGraph()
        for i in range((edgeList.size())[2]):
            G.add_edge(edgeList[j, 0, i], edgeList[j, 1, i])

        A = torch.from_numpy(networkx.to_numpy_array(G))
        #print("A get_edges \n", A.size() , "\n", A)

    # convert to numpy
    edge_list_np=edge_list[:, :, :].cpu().detach().numpy()
    #print(' numpy get_edges \n', edge_list_np.shape,  "\n", edge_list_np)

    #list of wheref the idx changes
    idx_list=[]
    for batch in range(batch_size):
        idx_list.append(np.where(edge_list_np[batch][0][:-1] != edge_list_np[batch][0][1:])[0]+1)

    idx_change = np.array(list(idx_list))
    #print("idx get_edges\n" ,idx_change.shape, "\n", idx_change)

    #k_ef=idx.max (della differenza tra indici)
    #if idx[:][:]<k_ef


    range_tensor=torch.range(0, num_points, device=features.device).repeat_interleave(k_ef)
    #print("range new get_edges\n" ,range_tensor.size(), "\n", range_tensor)

    idx_base=torch.range(0, num_points*batch_size, batch_size).repeat_interleave(num_points)
    #print("idx_base new get_edges\n" ,idx_base.size(), "\n", idx_base)


    #for i in range(num_points):

        #print("where\n", np.where(edge_list[:,0,:]==i))

        #edge_list[:,0,i]

    edge_list_tot=[]
    for batch in range(batch_size):
        for p, i in enumerate(edge_list_np[batch][0]):
            for j in range_tensor:
                if i==j:
                    edge_list_tot.append(edge_list_np[batch][1][p])
                else:
                    edge_list_tot.append(num_points+1)

    #print("edge list tot \n", len(edge_list_tot), '\n', edge_list_tot)


    #edge_list[:, 0, :]

    #edge_list_np[:, 0, idx]


    #edge_list_tot=  (batch_size*num_points=55*k_ef)

"""


# v1 is faster on GPU
def get_graph_edge_feature_v1(x, k, idx, ef):
    batch_size, num_dims, num_points = x.size()
    #print("x get graph\n ", x.size(),  "\n"  , x)
    #print("edge_features get graph\n ", ef.size(),  "\n"  , ef)

    #print("idx get graph\n ", idx.size(),  "\n"  , idx)  # (batch_size, num_points, k)
    idx_base = torch.arange(0, batch_size, device=x.device).view(-1, 1, 1) * num_points # (batch_size, 1, 1)
    #print("idx_base get graph\n ", idx_base.size(),  "\n"  , idx_base)
    idx = idx + idx_base # for each batch increases the index of a quantity equal to num_points
    #print("idx new get graph\n ", idx.size(),  "\n"  , idx)
    idx = idx.view(-1) # (batch_size*num_points*k)
    #print("idx new new get graph\n ", idx.size(),  "\n"  , idx)


    fts = x.transpose(2, 1).reshape(-1, num_dims)  # (batch_size, num_dims, num_points)-> (batch_size, num_points, num_dims) -> (batch_size*num_points, num_dims)
    #print("fts  get graph\n ", fts.size(),  "\n"  , fts)
    fts = fts[idx, :].view(batch_size, num_points, k, num_dims)  # neighbors: -> (batch_size*num_points*k, num_dims) -> ...
    #print("fts new get graph\n ", fts.size(),  "\n"  , fts)
    fts = fts.permute(0, 3, 1, 2).contiguous()  # (batch_size, num_dims, num_points, k)
    #print("fts new new get graph\n ", fts.size(),  "\n"  , fts)
    x = x.view(batch_size, num_dims, num_points, 1).repeat(1, 1, 1, k)
    #print("x new get graph\n ", x.size(),  "\n"  , x)



    fts = torch.cat((x, fts - x, ef), dim=1)  # ->(batch_size, 2*num_dims+num_ef, num_points, k)
    #print("fts last get graph\n ", fts.size(),  "\n"  , fts)

    return fts


# v2 is faster on CPU
def get_graph_edge_feature_v2(x, idx):
    batch_size, num_dims, num_points = x.size()

    idx_base = torch.arange(0, batch_size, device=x.device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)

    fts = x.transpose(0, 1).reshape(num_dims, -1)  # -> (num_dims, batch_size, num_points) -> (num_dims, batch_size*num_points)
    fts = fts[:, idx].view(num_dims, batch_size, num_points, k)  # neighbors: -> (num_dims, batch_size*num_points*k) -> ...
    fts = fts.transpose(1, 0).contiguous()  # (batch_size, num_dims, num_points, k)

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
        self.get_graph_feature = get_graph_feature_v2 if cpu_mode else get_graph_feature_v1

        self.convs = nn.ModuleList()
        for i in range(self.num_layers):
            self.convs.append(nn.Conv2d(2 * in_feat if i == 0 else out_feats[i - 1], out_feats[i], kernel_size=1, bias=False if self.batch_norm else True))

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


class EdgeFeatureConvBlock(nn.Module):
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

    def __init__(self, idx, k, k_ef, in_feat, out_feats, in_edgefeat, out_edgefeats, batch_norm=True, activation=True, cpu_mode=False):
        super(EdgeFeatureConvBlock, self).__init__()
        self.batch_norm = batch_norm
        self.activation = activation
        self.num_layers = len(out_feats)
        self.num_layers_ef = len(out_edgefeats)
        self.k = k
        self.k_ef = k_ef
        self.get_graph_edge_feature = get_graph_edge_feature_v2 if cpu_mode else get_graph_edge_feature_v1
        self.get_graph_feature = get_graph_feature_v2 if cpu_mode else get_graph_feature_v1

        self.convs = nn.ModuleList()
        for i in range(self.num_layers):
            in_feat_tot=2 * in_feat + in_edgefeat if EDGE_FEATURES_OR_CONNECTION and idx==0 else 2 * in_feat
            self.convs.append(nn.Conv2d(in_feat_tot if i == 0 else out_feats[i - 1],
                                        out_feats[i] , kernel_size=1, bias=False if self.batch_norm else True))


        self.convs_ef = nn.ModuleList()
        for i in range(self.num_layers_ef):
            self.convs_ef.append(nn.Conv2d(in_edgefeat if i == 0 else out_edgefeats[i - 1],
                                           out_edgefeats[i], kernel_size=1, bias=False if self.batch_norm else True))


        if batch_norm:
            self.bns = nn.ModuleList()
            for i in range(self.num_layers):
                self.bns.append(nn.BatchNorm2d(out_feats[i]))
            self.bns_ef = nn.ModuleList()
            for i in range(self.num_layers_ef):
                self.bns_ef.append(nn.BatchNorm2d(out_edgefeats[i]))

        if activation:
            self.acts = nn.ModuleList()
            for i in range(self.num_layers):
                self.acts.append(nn.ReLU())
            self.acts_ef = nn.ModuleList()
            for i in range(self.num_layers_ef):
                self.acts_ef.append(nn.ReLU())

        if in_feat == out_feats[-1]:
            self.sc = None
        else:
            self.sc = nn.Conv1d(in_feat, out_feats[-1], kernel_size=1, bias=False)
            self.sc_bn = nn.BatchNorm1d(out_feats[-1])

        if in_edgefeat == out_edgefeats[-1]:
            self.sc_ef = None
        else:
            self.sc_ef = nn.Conv1d(in_edgefeat, out_edgefeats[-1], kernel_size=1, bias=False)
            self.sc_bn_ef = nn.BatchNorm1d(out_edgefeats[-1])

        if activation:
            self.sc_act = nn.ReLU()
            self.sc_act_ef = nn.ReLU()

    def forward(self, points, features, edge_list, edge_features, idx):
        #print('edge_features block:\n', edge_features.size(), '\n', edge_features) #(batch_size, num_ef=28, dim_edgefeatures=625)
        #print("features block \n ",features.size(),  "\n", features )# size (batch_size, num_dims=32, num_points=55)
        #print("points block \n ",points.size(),  "\n" ) # size (batch_size, 2, num_points)

        batch_size, num_dims, num_points = features.size()
        ef_sparse_tensor=build_sparse_tensor(edge_features, edge_list, num_points)

        if NEW_EDGES:
            topk_indices = get_edges(edge_list, batch_size, num_points, self.k_ef) #(batch_size, 2, dim_edgefeatures)
        elif EDGE_FEATURES_OR_CONNECTION and idx==0:
            topk_indices = knn(points, self.k)
            topk_indices, edge_features = edges_or_new(topk_indices, edge_list, batch_size, num_points, self.k_ef, edge_features)
        else:
            topk_indices = knn(points, self.k)

        #print("topk_indices block \n ", topk_indices, "\n", topk_indices.size(), "\n" ) # size(batch_size, num_points, k)
        #print("topk_indices block \n ", topk_indices)
        #print("edge_indices block \n ", edge_indices.size(),  "\n"  , edge_indices)

        if EDGE_FEATURES_OR_CONNECTION and idx==0:
            x = self.get_graph_edge_feature(features, self.k, topk_indices, edge_features)
        else:
            x = self.get_graph_feature(features, self.k, topk_indices)

        #print("x    \n", x.size(), "\n" )

        for conv, bn, act in zip(self.convs, self.bns, self.acts):
            x = conv(x)  # (N, C', P, K) = (batch_size, num_dims, num_points, k-nn)
            #print("x 1   \n", x.size(), "\n" )
            if bn:
                x = bn(x)
                #print("x  2  \n", x.size(), "\n" )
            if act:
                x = act(x)
                #print("x  3  \n", x.size(), "\n" )


        fts = x.mean(dim=-1)  # (N, C, P)
        #print("fts block \n ", fts.size(),  "\n"  , fts)

        # shortcut
        if self.sc:
            sc = self.sc(features)  # (N, C_out, P)
            sc = self.sc_bn(sc)
        else:
            sc = features


        if EDGE_FEATURES:
            for convs_ef, bn_ef, act_ef in zip(self.convs_ef, self.bns_ef, self.acts_ef):
                edge_features = convs_ef(edge_features)
                if bn_ef:
                    edge_features = bn_ef(edge_features)
                if act_ef:
                    edge_features = act(edge_features)

            edge_fts = edge_features.mean(dim=-1)  # (N, C, P)

            # shortcut
            if self.sc_ef:
                sc_ef = self.sc_ef(edge_fts)  # (N, C_out, P)
                sc_ef = self.sc_bn_ef(sc_ef)
            else:
                sc_ef = edge_fts

        ## num points deve esser euguale per fts e edge_fts per poterle concatenare così!!!
            return torch.cat((self.sc_act(sc + fts), self.sc_act_ef(sc_ef + edge_fts) ), dim=1) # (N, C_out, P)

        #print("sc_act block \n ", self.sc_act(sc + fts).size(),  "\n"  , self.sc_act(sc + fts))
        return self.sc_act(sc + fts) # (N, C_out, P)



class ParticleEdgeNet(nn.Module):

    def __init__(self,
                 input_dims,
                 edge_features_dims,
                 num_classes,
                 conv_params=[(7, (32, 32, 32)), (7, (64, 64, 64))],
                 conv_params_ef=[(7, (32, 32, 32)), (7, (64, 64, 64))],
                 fc_params=[(128, 0.1)],
                 use_fusion=True,
                 use_fts_bn=True,
                 use_counts=True,
                 for_inference=False,
                 for_segmentation=False,
                 **kwargs):
        super(ParticleEdgeNet, self).__init__(**kwargs)

        self.use_fts_bn = use_fts_bn
        if self.use_fts_bn:
            self.bn_fts = nn.BatchNorm1d(input_dims)

        self.use_counts = use_counts

        self.k_ef = conv_params_ef[0][0]

        self.edge_feat_convs = nn.ModuleList()
        for idx, (layer_param, layer_param_ef) in enumerate(zip(conv_params, conv_params_ef)):
            k, channels = layer_param
            k_ef, channels_ef = layer_param_ef
            in_feat = input_dims if idx == 0 else conv_params[idx - 1][1][-1]
            in_edgefeat = edge_features_dims if idx == 0 else conv_params_ef[idx - 1][1][-1]
            self.edge_feat_convs.append(EdgeFeatureConvBlock(idx, k=k, k_ef=k_ef,
                                        in_feat=in_feat, out_feats=channels,
                                        in_edgefeat=in_edgefeat, out_edgefeats=channels_ef,
                                        cpu_mode=for_inference))
        self.use_fusion = use_fusion
        if self.use_fusion:
            in_chn = sum(x[-1] for _, x in conv_params)
            out_chn = np.clip((in_chn // 128) * 128, 128, 1024)
            self.fusion_block = nn.Sequential(nn.Conv1d(in_chn, out_chn, kernel_size=1, bias=False), nn.BatchNorm1d(out_chn), nn.ReLU())

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

    def forward(self, points, features, edge_list, edge_features, mask=None):
        #print('points net:\n', points.size())
        #print('features net:\n', features.size())
        #print('edge_features net:\n', edge_features.size())

        if mask is None:
            mask = (features.abs().sum(dim=1, keepdim=True) != 0)  # (N, 1, P)
        #print ('mask \n', mask)
        points *= mask
        #print('points net:\n', points)

        features *= mask
        batch_size, num_dims, num_points = features.size()


        #reshape fts and pts and mask
        if NEW_EDGES_DUMMY or EDGE_FEATURES_OR_CONNECTION_DUMMY:
            dummy_mask= torch.zeros(batch_size, 1, self.k_ef, device=mask.device)
            mask=torch.cat((mask, dummy_mask), dim=2)
            dummy_features= torch.zeros(batch_size, num_dims, self.k_ef, device=features.device)
            features=torch.cat((features, dummy_features), dim=2)
            dummy_pts= torch.zeros(batch_size, 2, self.k_ef, device=points.device)
            points=torch.cat((points, dummy_pts), dim=2)

        #reshape fts and pts and mask
        if NEW_EDGES or EDGE_FEATURES_OR_CONNECTION:
            dummy_mask= torch.zeros(batch_size, 1, 1, device=mask.device)
            mask=torch.cat((mask, dummy_mask), dim=2)
            dummy_features= torch.zeros(batch_size, num_dims, 1, device=features.device)
            features=torch.cat((features, dummy_features), dim=2)
            dummy_pts= torch.zeros(batch_size, 2, 1, device=points.device)
            points=torch.cat((points, dummy_pts), dim=2)

        coord_shift = (mask == 0) * 1e9 # if masked add 1e9 to coordinates and are not considered in the clustering to knn
        if self.use_counts:
            counts = mask.float().sum(dim=-1)
            counts = torch.max(counts, torch.ones_like(counts))  # >=1

        if self.use_fts_bn:
            fts = self.bn_fts(features) * mask
        else:
            fts = features
        outputs = []



        for idx, conv in enumerate(self.edge_feat_convs):
            pts = (points if idx == 0 else fts) + coord_shift
            #print('points net:\n', pts)

            fts = conv(pts, fts, edge_list, edge_features, idx) * mask
            if self.use_fusion:
                outputs.append(fts)
            #break
        if self.use_fusion:
            fts = self.fusion_block(torch.cat(outputs, dim=1)) * mask

#         assert(((fts.abs().sum(dim=1, keepdim=True) != 0).float() - mask.float()).abs().sum().item() == 0)

        if self.for_segmentation:
            x = fts
        else:
            if self.use_counts:
                x = fts.sum(dim=-1) / counts  # divide by the real counts
            else:
                x = fts.mean(dim=-1)

        #output = x
        output = self.fc(x)
        if self.for_inference:
            output = torch.softmax(output, dim=1)
        #print('output:\n', output)
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



class ParticleEdgeNetTagger(nn.Module):

    def __init__(self,
                 pf_features_dims,
                 sv_features_dims,
                 edge_features_dims,
                 num_classes,
                 conv_params=[(7, (32, 32, 32)), (7, (64, 64, 64))],
                 conv_params_ef=[(7, (32, 32, 32)), (7, (64, 64, 64))],
                 fc_params=[(128, 0.1)],
                 use_fusion=True,
                 use_fts_bn=True,
                 use_counts=True,
                 pf_input_dropout=None,
                 sv_input_dropout=None,
                 for_inference=False,
                 **kwargs):
        super(ParticleEdgeNetTagger, self).__init__(**kwargs)
        self.pf_input_dropout = nn.Dropout(pf_input_dropout) if pf_input_dropout else None
        self.sv_input_dropout = nn.Dropout(sv_input_dropout) if sv_input_dropout else None
        self.pf_conv = FeatureConv(pf_features_dims, 32) # sets the dimension of the features of pv and pf to be equal
        self.sv_conv = FeatureConv(sv_features_dims, 32)
        #self.ef_conv = FeatureConv(edge_features_dims, 32)
        self.pn = ParticleEdgeNet(input_dims=32,
                              edge_features_dims=edge_features_dims,
                              num_classes=num_classes,
                              conv_params=conv_params,
                              conv_params_ef=conv_params_ef,
                              fc_params=fc_params,
                              use_fusion=use_fusion,
                              use_fts_bn=use_fts_bn,
                              use_counts=use_counts,
                              for_inference=for_inference)
    def forward(self, pf_points, pf_features, pf_mask, sv_points, sv_features, sv_mask, track_ef_idx, track_ef):

        if self.pf_input_dropout:
            pf_mask = (self.pf_input_dropout(pf_mask) != 0).float()
            pf_points *= pf_mask
            pf_features *= pf_mask
        if self.sv_input_dropout:
            sv_mask = (self.sv_input_dropout(sv_mask) != 0).float()
            sv_points *= sv_mask
            sv_features *= sv_mask

        points = torch.cat((pf_points, sv_points), dim=2)
        #print("points tagger \n ",points.size(),  "\n" , points) # size (batch_size, 2, num_points)

        features = torch.cat((self.pf_conv(pf_features * pf_mask) * pf_mask, self.sv_conv(sv_features * sv_mask) * sv_mask), dim=2)
        mask = torch.cat((pf_mask, sv_mask), dim=2)

        #edge_features= self.ef_conv(edge_features* edge_features_mask)*edge_features_mask
        #print('features tagger:\n', features.size())

        return self.pn(points, features, track_ef_idx, track_ef, mask) # call the forward of particle net
