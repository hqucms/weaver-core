import torch
from weaver.utils.logger import _logger
from weaver.nn.model.ParticleEdgeNeXt import ParticleEdgeNeXtTagger


def get_model(data_config, **kwargs):
    cfg = dict(
        pf_features_dims=len(data_config.input_dicts['pf_features'])  if 'pf_features' in data_config.input_dicts else 0,
        sv_features_dims = len(data_config.input_dicts['sv_features']) if 'sv_features' in data_config.input_dicts else 0,
        edge_input_dim=len(data_config.input_dicts['track_ef']) if 'track_ef' in data_config.input_dicts else 0,
        num_classes=len(data_config.label_value),
        num_aux_classes_clas=len([k for k in data_config.aux_label_value_clas if 'clas' in k]),
        num_aux_classes_regr=len([k for k in data_config.aux_label_value_regr if 'regr' in k]),
        num_aux_classes_pair=len([k for k in data_config.aux_label_value_pair if 'bin' in k]),
        # network configurations
        node_dim=32,
        edge_dim=24,
        use_edge_bn=True,
        layer_params=[(16, 256, [(4, 1), (2, 1), (1, 1)], 64), (16, 256, [(4, 1), (2, 1), (1, 1)], 64), (16, 256, [(4, 1), (2, 1), (1, 1)], 64)],  # noqa
        fc_params=[(256, 0.1)],
        global_aggregation='attn4',
        # MultiScaleEdgeConv options
        edge_aggregation='attn8',
        use_rel_lv_fts=True, #False
        use_polarization_angle=False,
        use_rel_fts=False,
        use_rel_dist=False,
        update_coords=False,
        lv_aggregation=False,
        use_node_se=True,
        use_edge_se=True,
        init_scale=1e-5,
    )
    cfg.update(**kwargs)
    _logger.info('Model config: %s' % str(cfg))

    model = ParticleEdgeNeXtTagger(**cfg)

    model_info = {
        'input_names': list(data_config.input_names),
        'input_shapes': {k: ((1,) + s[1:]) for k, s in data_config.input_shapes.items()},
        'output_names': ['softmax'],
        'dynamic_axes': {**{k: {0: 'N', 2: 'n_' + k.split('_')[0]} for k in data_config.input_names}, **{'softmax': {0: 'N'}}},
    }

    return model, model_info


def get_loss(data_config, **kwargs):
    return torch.nn.CrossEntropyLoss()

def get_aux_loss_clas(data_config, dev, **kwargs):
    return torch.nn.CrossEntropyLoss(weight=torch.FloatTensor([5, 10, 40, 1.5]).to(dev))

def get_aux_loss_regr(data_config, **kwargs):
    return torch.nn.MSELoss()

def get_aux_loss_bin(data_config, **kwargs):
    return torch.nn.BCEWithLogitsLoss()
