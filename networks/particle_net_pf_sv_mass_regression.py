import torch
from torch import Tensor

from utils.nn.model.ParticleNet import ParticleNetTagger


def get_model(data_config, **kwargs):
    conv_params = [
        (16, (64, 64, 64)),
        (16, (128, 128, 128)),
        (16, (256, 256, 256)),
        ]
    fc_params = [(256, 0.1)]
    use_fusion = True

    pf_features_dims = len(data_config.input_dicts['pf_features'])
    sv_features_dims = len(data_config.input_dicts['sv_features'])
    num_classes = 1
    model = ParticleNetTagger(pf_features_dims, sv_features_dims, num_classes,
                              conv_params, fc_params,
                              use_fusion=use_fusion,
                              use_fts_bn=kwargs.get('use_fts_bn', False),
                              use_counts=kwargs.get('use_counts', True),
                              pf_input_dropout=kwargs.get('pf_input_dropout', None),
                              sv_input_dropout=kwargs.get('sv_input_dropout', None),
                              for_inference=False,
                              )

    model_info = {
        'input_names': list(data_config.input_names),
        'input_shapes': {k: ((1,) + s[1:]) for k, s in data_config.input_shapes.items()},
        'output_names': ['output'],
        'dynamic_axes': {**{k: {0: 'N', 2: 'n_' + k.split('_')[0]} for k in data_config.input_names}, **{'output': {0: 'N'}}},
        }

    return model, model_info


class RatioSmoothL1Loss(torch.nn.SmoothL1Loss):
    __constants__ = ['reduction']

    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean', beta: float = 1.0, cutoff: float = 1.0) -> None:
        super(RatioSmoothL1Loss, self).__init__(size_average, reduce, reduction)
        self.beta = beta
        self.cutoff = cutoff

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        ratio = input / torch.maximum(target, self.cutoff * torch.ones_like(target))
        return torch.nn.functional.smooth_l1_loss(ratio, torch.ones_like(target), reduction=self.reduction, beta=self.beta)


def get_loss(data_config, **kwargs):
    if kwargs.get('loss_mode', 0) == 1:
        return RatioSmoothL1Loss(beta=kwargs.get('loss_beta', 0.3), cutoff=kwargs.get('loss_cutoff', 1))
    else:
        return torch.nn.SmoothL1Loss(beta=kwargs.get('loss_beta', 10))
