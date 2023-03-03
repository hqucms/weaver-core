import os
import traceback
from functools import partial

import numpy as np
import sklearn.metrics as _m

from ..logger import _logger

np.set_printoptions(threshold=np.inf)

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


def confusion_matrix(y_true, y_score, aux_type=None):
    if y_true is None or y_score is None:
        return None
    if isinstance(y_true,dict):
        y_true = y_true[aux_type]
    if y_score.ndim == 1:
        y_pred = y_score > 0.5
        #print('y_score', y_score )
        #print('y_pred', y_pred )
    else:
        y_pred = y_score.argmax(1)
    return _m.confusion_matrix(y_true, y_pred, normalize='true')


def save_labels(y_true, y_score, epoch, roc_prefix, label_type):
    if y_true is None or y_score is None:
        return None

    y_mask_from_b= None
    outfile=f'{roc_prefix}{label_type}_labels_epoch_{epoch:02d}.npz'
    if isinstance(y_true,dict):
        if label_type == 'pf_regr':
            try:
                y_mask_from_b = y_true['pf_regr_mask_from_b']
            except:
                pass
        y_true = y_true[label_type]
    with open(outfile, 'wb') as f:
        np.savez(f, y_true=y_true, y_score=y_score, y_mask_from_b = y_mask_from_b)

    return f'y_true and y_score {label_type} for epoch {epoch} properly saved in file: \n {outfile}\n'


_metric_dict = {
    'roc_auc_score': partial(_m.roc_auc_score, multi_class='ovo'),
    'roc_auc_score_matrix': roc_auc_score_ovo,
    'confusion_matrix': confusion_matrix,
    'save_labels': save_labels,
    'aux_confusion_matrix_pf_clas': confusion_matrix,
    'aux_confusion_matrix_pair_bin': confusion_matrix,
    'aux_save_labels_pf_clas': save_labels,
    'aux_save_labels_pf_regr': save_labels,
    'aux_save_labels_pair_bin': save_labels,
}


def _get_metric(metric):
    try:
        return _metric_dict[metric]
    except KeyError:
        return getattr(_m, metric)


def evaluate_metrics(y_true, y_score, aux_y_true, aux_y_score_pf_clas, aux_y_score_pf_regr, aux_y_score_pair_bin,  eval_metrics=[], eval_aux_metrics=[], epoch=-1, roc_prefix=None):
    results = {}
    for metric in eval_metrics:
        func = _get_metric(metric)
        try:
            results[metric] = func(y_true, y_score, epoch, roc_prefix, 'primary') if 'label' in metric else func(y_true, y_score)
        except Exception as e:
            results[metric] = None
            _logger.error(str(e))
            _logger.debug(traceback.format_exc())

    aux_dict = {
        'pf_clas':aux_y_score_pf_clas,
        'pf_regr':aux_y_score_pf_regr,
        'pair_bin':aux_y_score_pair_bin,
    }

    for aux_metric in eval_aux_metrics:
        func = _get_metric(aux_metric)
        for k, v in aux_dict.items():
            if k in aux_metric:
                break
        try:
            results[aux_metric] = func(aux_y_true, v, epoch, roc_prefix, k) if 'label' in aux_metric else func(aux_y_true, v)
        except Exception as e:
            results[aux_metric] = None
            _logger.error(str(e))
            _logger.debug(traceback.format_exc())

    return results
