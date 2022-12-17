import numpy as np
import traceback
import sklearn.metrics as _m
from functools import partial
from ..logger import _logger
import torch

# def _bkg_rejection(y_true, y_score, sig_eff):
#     fpr, tpr, _ = _m.roc_curve(y_true, y_score)
#     idx = next(idx for idx, v in enumerate(tpr) if v > sig_eff)
#     rej = 1. / fpr[idx]
#     return rej
#
#
# def bkg_rejection(y_true, y_score, sig_eff):
#     if y_score.ndim == 1:
#         return _bkg_rejection(y_true, y_score, sig_eff)
#     else:
#         num_classes = y_score.shape[1]
#         for i in range(num_classes):
#             for j in range(i + 1, num_classes):
#                 weights = np.logical_or(y_true == i, y_true == j)
#                 truth =


def roc_auc_score_ovo(y_true, y_score):
    if y_score.ndim == 1:
        return _m.roc_auc_score(y_true, y_score)
    else:
        num_classes = y_score.shape[1]
        result = np.zeros((num_classes, num_classes), dtype='float32')
        for i in range(num_classes):
            for j in range(i + 1, num_classes):
                weights = np.logical_or(y_true == i, y_true == j)
                truth = y_true == j
                score = y_score[:, j] / np.maximum(y_score[:, i] + y_score[:, j], 1e-6)
                result[i, j] = _m.roc_auc_score(truth, score, sample_weight=weights)
    return result


def confusion_matrix(y_true, y_score):
    if y_score.ndim == 1:
        y_pred = y_score > 0.5
    else:
        y_pred = y_score.argmax(1)
    return _m.confusion_matrix(y_true, y_pred, normalize='true')

def roc_curve_bVSuds(y_true, y_score):
    y_true_b = y_true==0
    y_true_bb = y_true==1
    y_true_uds = y_true==4
    y_true_idx = torch.logical_or(torch.logical_or(y_true_b ,y_true_bb),y_true_uds)
    y_score_b=y_score[:,0]*y_true_b
    y_score_bb=y_score[:,1]*y_true_bb
    y_score_uds=(y_score[:,0]+y_score[:,1])*y_true_uds
    y_score_tot=y_score_b+y_score_bb+y_score_uds
    y_score_tot = y_score_tot[y_true_idx]
    y_true_tot=torch.ones_like(torch.logical_or(y_true_b ,y_true_bb), device=y_true.device)[y_true_idx]

    return _m.roc_curve(y_true_tot, y_score_tot)



_metric_dict = {
    'roc_auc_score': partial(_m.roc_auc_score, multi_class='ovo'),
    'roc_auc_score_matrix': roc_auc_score_ovo,
    'confusion_matrix': confusion_matrix,
    'roc_curve_bVSuds': roc_curve_bVSuds,
    }


def _get_metric(metric):
    try:
        return _metric_dict[metric]
    except KeyError:
        return getattr(_m, metric)


def evaluate_metrics(y_true, y_score, eval_metrics=[]):
    results = {}
    for metric in eval_metrics:
        func = _get_metric(metric)
        try:
            results[metric] = func(y_true, y_score)
        except Exception as e:
            results[metric] = None
            _logger.error(str(e))
            _logger.debug(traceback.format_exc())
    return results
