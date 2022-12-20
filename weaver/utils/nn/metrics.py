import os
import traceback
from functools import partial

#import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics as _m

from ..logger import _logger
#np.set_printoptions(threshold=np.inf)

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


def get_labels(y_true, y_score, labels_s, labels_b):
    #print('y_true   ', y_true, '\n y_score  ', y_score, '\n')
    y_true_s = np.logical_or.reduce([y_true==label for label in labels_s])
    y_true_b = np.logical_or.reduce([y_true==label for label in labels_b])
    y_true_idx = np.logical_or(y_true_s,y_true_b)
    y_score_s=sum([y_score[:,label] for label in labels_s])*y_true_s
    y_score_b=sum([y_score[:,label] for label in labels_s])*y_true_b
    #print('y_true_s   ', y_true_s, '\n y_score_s  ', y_score_s,'\ny_true_b   ', y_true_b, '\n y_score_b  ', y_score_b, '\n y_true_idx', y_true_idx)
    y_score_tot=y_score_s+y_score_b
    y_score_tot = y_score_tot[y_true_idx]
    y_true_tot=y_true_s[y_true_idx].astype(int)

    return y_true_tot, y_score_tot

def roc_curve_bVSuds(y_true, y_score, epoch,roc_prefix):
    y_true_tot, y_score_tot = get_labels(y_true, y_score, [0,1], [4])
    #print('y_true_uds   ', y_true_tot, '\n y_score_uds  ', y_score_tot, '\n')

    '''
    _m.RocCurveDisplay.from_predictions(y_true_tot, y_score_tot, name=f'b vs uds', color='darkorange')

    plt.plot([0, 1], [0, 1], 'k--', label='chance level (AUC = 0.5)')
    plt.axis('square')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'b VS uds for epoch #{epoch}')
    plt.legend()
    plt.savefig(os.path.join(roc_prefix,f'roc_curve_bVSuds_#{epoch}.png'))
    '''

    outfile=f'{roc_prefix}_y_bVSuds_epoch{epoch}.npy'
    with open(outfile, 'wb') as f:
        np.save(f, np.array([y_true_tot, y_score_tot]))

    return f'y_true and y_score for b VS uds for epoch {epoch} properly saved in file: \n {outfile}\n'


def roc_curve_bVSg(y_true, y_score, epoch,roc_prefix):
    y_true_tot, y_score_tot = get_labels(y_true, y_score, [0,1], [5])
    #print('y_true_g   ', y_true_tot, '\n y_score_g  ', y_score_tot, '\n')


    '''
    _m.RocCurveDisplay.from_predictions(y_true_tot, y_score_tot, name=f'b vs g', color='darkorange')

    plt.plot([0, 1], [0, 1], 'k--', label='chance level (AUC = 0.5)')
    plt.axis('square')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'b VS g for epoch #{epoch}')
    plt.legend()
    plt.savefig(os.path.join(roc_prefix,f'roc_curve_bVSg_#{epoch}.png'))
    '''
    outfile=f'{roc_prefix}_y_bVSg_epoch{epoch}.npy'
    with open(outfile, 'wb') as f:
        np.save(f, np.array([y_true_tot, y_score_tot]))

    return f'y_true and y_score for b VS g for epoch {epoch} properly saved in file: \n {outfile}\n'

_metric_dict = {
    'roc_auc_score': partial(_m.roc_auc_score, multi_class='ovo'),
    'roc_auc_score_matrix': roc_auc_score_ovo,
    'confusion_matrix': confusion_matrix,
    'roc_curve_bVSuds': roc_curve_bVSuds,
    'roc_curve_bVSg': roc_curve_bVSg,
    }


def _get_metric(metric):
    try:
        return _metric_dict[metric]
    except KeyError:
        return getattr(_m, metric)


def evaluate_metrics(y_true, y_score, eval_metrics=[], epoch=-1, roc_prefix=None):
    results = {}
    for metric in eval_metrics:
        func = _get_metric(metric)
        try:
            results[metric] = func(y_true, y_score, epoch, roc_prefix) if 'roc_curve' in metric else func(y_true, y_score)
        except Exception as e:
            results[metric] = None
            _logger.error(str(e))
            _logger.debug(traceback.format_exc())
    return results
