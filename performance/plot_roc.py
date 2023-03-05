import numpy as np
import sklearn.metrics as _m
import matplotlib.pyplot as plt
import os
import sys
import re
import pickle
import argparse
from collections import defaultdict, OrderedDict
import mplhep as hep
import yaml
import time


orig_stdout = sys.stdout
f = open('roc.txt', 'w')
sys.stdout = f

np.set_printoptions(threshold=np.inf)

# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=str, default='-1',
                    help='roc for various epochs')
parser.add_argument('--not-show', action='store_true', default=False,
                    help='do not show plots')
parser.add_argument('--save', action='store_true', default=False,
                    help='save plots')
parser.add_argument('--only-primary', action='store_true', default=False,
                    help='only compute the primary ROC')
parser.add_argument('--history', action='store_true', default=False,
                    help='only compute the primary ROC')
parser.add_argument('--path', type=str, default='',
                    help='input path')
parser.add_argument('--name', type=str, default='',
                    help='name of the configuration')
parser.add_argument('--in-dict', type=str, default='performance_comparison',
                    help='name of the file with the dictionary')
args = parser.parse_args()

# dictionary where the first key is the epoch,
# the second is the type of ROC
# and the third is the network
epochs_dict=defaultdict(lambda: defaultdict(defaultdict))

# dictionary with the labels for the ROC curves
roc_type_dict=OrderedDict([
    ('pair_bin',{
        'PAIRbin' : [[0], None]
    }),
    ('pf_clas',{
        'PFclas_b+bcVSc+other' : [[0,2], [1,3]]
    }),
    ('pf_regr',{
        'PFregr' : [None, None]
    }),
    ('primary',{
        'JET_bVSuds':[[0,1], [4]],
        'JET_bVSg':[[0,1], [5]],
        'JET_bVSudsg':[[0,1], [4,5]]
    }),
])

# dictionary with the axes limits
axis_limits ={
    0: ((600, 600),(0,3,0,0.5), '_vtx_dist_pv'),
    1: ((300, 300),(-0.15,0.15,-0.05,0.05), '_vtx_x'),
    2: ((300, 300),(-0.15,0.15,-0.1,0.1), '_vtx_y'),
    3: ((600, 600),(-0.25, 0.25,-2,2), '_vtx_z')
}

# get the labels for the ROC curves
def get_labels(y_true, y_score, labels_s, labels_b):
    if labels_b is None:
        y_true_tot = y_true
        y_score_tot = y_score
    else:
        y_true_s = np.logical_or.reduce([y_true==label for label in labels_s])
        y_true_b = np.logical_or.reduce([y_true==label for label in labels_b])
        y_true_idx = np.logical_or(y_true_s,y_true_b)
        y_score_s=sum([y_score[:,label] for label in labels_s])*y_true_s
        y_score_b=sum([y_score[:,label] for label in labels_s])*y_true_b
        y_score_tot=y_score_s+y_score_b
        y_score_tot = y_score_tot[y_true_idx]
        y_true_tot=y_true_s[y_true_idx].astype(int)

    return y_true_tot, y_score_tot

# compute the ROC curve and the AUC
def get_rates(y_t, y_s, l_s, l_b):
    if l_s is None and l_b is None:
        fpr, tpr, roc_auc = y_s, y_t, np.nan
    else:
        y_true, y_score = get_labels(y_t,  y_s, l_s, l_b)
        fpr, tpr, thresh = _m.roc_curve(y_true, y_score)
        roc_auc = _m.roc_auc_score(y_true, y_score)
    return fpr, tpr, roc_auc

def plt_fts(out_dir, roc_type, name, fig_handle, axis_lim=None):
    if 'regr' in roc_type:
        #plt.axis('square')
        plt.xlabel('True-Reco')
        if 'True-Reco' not in name:
            plt.xlim([axis_lim[0], axis_lim[1]])
            plt.xlabel('True')
            plt.ylabel('Reco')
            plt.plot([-40, 40], [-40, 40], 'y--', label='True = Reco')
            plt.ylim([axis_lim[2], axis_lim[3]])
    else:
        plt.xlabel('Efficency for b-jet (TP)')
        plt.ylabel('Mistagging prob (FP)')
        plt.ylim([1e-4, 1.005])
        plt.xlim([0.5, 1.0005])
        plt.yscale('log')

    plt.grid()

    hep.style.use('CMS')
    hep.cms.label(rlabel='')
    hep.cms.lumitext(f'ROC_{roc_type}_{name}')

    plt.legend(labelcolor='linecolor')
    #plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    fig_handle.set_size_inches(20, 15)
    plt.savefig(f'{out_dir}/roc_{roc_type}_{name}.png', bbox_inches='tight')
    if args.save:
        with open(f'{out_dir}/roc_{roc_type}_{name}.pickle', 'wb') as f:
            pickle.dump(fig_handle, f)
    if not args.not_show:
        plt.show()
    plt.close()

def load_dict(name):
    with open(name, 'r') as stream:
        loaded_dict=yaml.safe_load(stream)
    info_dict = defaultdict(list)
    for k, v in loaded_dict.items():
        info_dict[k].append(defaultdict(list))
        info_dict[k].append(defaultdict(list))
        info_dict[k].append(v[0])
        info_dict[k].append(v[1])
    return info_dict

if __name__ == '__main__':
    label_dict=load_dict(f'{args.in_dict}.yaml')

    # create output directory
    date_time = time.strftime('%Y%m%d-%H%M%S')
    out_dir = os.path.join(f'{args.path}roc_curve', f'{date_time}_{args.name}_roc')
    os.makedirs(out_dir, exist_ok=True)

    for input_name, info in label_dict.items():
        # files to load
        dir_name=f'{args.path}{input_name}'
        files = [filename for filename in os.listdir(dir_name)
                 if ('labels_epoch' in filename)]
        best_files = [filename for filename in os.listdir(dir_name)
                    if ('labels_best' in filename)]

        # epochs to load
        if args.epochs == '-1':
            epoch_list=[len([k for k in files if 'primary' in k])-1]
        elif args.epochs:
            epoch_list=[int(i) for i in args.epochs.split(',')]
        else:
            epoch_list = []

        # load files for each epoch
        for infile in files:
            epoch = int(infile.split('.npz')[0][-2:] if infile.split('.npz')[0][-2].isnumeric() else infile.split('.npz')[0][-1])
            if epoch in epoch_list:
                for label_type, labels_info in roc_type_dict.items():
                    if args.only_primary and label_type != 'primary':
                        continue
                    if label_type in infile:
                        print(infile)
                        # load labels for each epoch and label type
                        with open(os.path.join(dir_name,infile), 'rb') as f:
                            y_true = np.load(f, allow_pickle=False)['y_true']
                            y_score = np.load(f, allow_pickle=False)['y_score']
                            # load the mask for the labels (if present)
                            try:
                                y_mask=np.load(f, allow_pickle=False)['y_mask_from_b']
                                y_true=y_true[y_mask]
                                y_score=y_score[y_mask]
                                print(f'MASK FOUND in {infile}! \n')
                            except KeyError:
                                pass
                            info[0][label_type].append(y_true)
                            info[1][label_type].append(y_score)

        # load files of best epoch
        for best_file in best_files:
            for label_type, labels_info in roc_type_dict.items():
                if args.only_primary and label_type != 'primary':
                    continue
                if label_type in best_file:
                    print(best_file)
                    # load labels for best epoch and for every label type
                    with open(os.path.join(dir_name,best_file), 'rb') as f:
                        y_true_best=np.load(f, allow_pickle=False)['y_true']
                        y_score_best=np.load(f, allow_pickle=False)['y_score']
                        print('y_true_best', y_true_best)
                        # load the mask for the labels (if present)
                        try:
                            y_mask=np.load(f, allow_pickle=False)['y_mask_from_b']
                            print('y_mask', y_mask)
                            y_true_best=y_true_best[y_mask]
                            y_score_best=y_score_best[y_mask]
                            print('y_true_best_masked', y_true_best)
                            print(f'MASK FOUND in {best_file}! \n')
                        except KeyError:
                            pass
                        # compute roc curve for best epoch
                        for roc_type, labels in labels_info.items():
                            fpr, tpr, roc_auc=get_rates(y_true_best,y_score_best,
                                                        labels[0], labels[1])
                            epochs_dict['best'][roc_type][info[2]]=(fpr, tpr, roc_auc, info[3])

        # compute roc curve for each epoch
        for label_type, labels_info in roc_type_dict.items():
            if len(info[0][label_type]) == 0: continue
            for roc_type, labels in labels_info.items():
                if 'regr' not in roc_type and args.history:
                    fig_handle = plt.figure()
                # loop over epochs
                for num in range(len(info[0][label_type])):
                    #print(roc_type, label_type,labels, epoch_list[num])

                    # compute roc curve for each epoch
                    fpr, tpr, roc_auc=get_rates(
                        info[0][label_type][num],info[1][label_type][num],
                        labels[0], labels[1])
                    # save roc curve for each epoch in a dictionary
                    epochs_dict[epoch_list[num]][roc_type][info[2]]=(fpr, tpr, roc_auc, info[3])
                    if 'regr' not in roc_type and args.history:
                        plt.plot(tpr,fpr,label=f'ROC {roc_type} {info[2]} epoch #{epoch_list[num]}, auc=%0.3f'% roc_auc)
                if 'regr' not in roc_type and args.history:
                    plt_fts(out_dir, roc_type, f'{info[2]}_history', fig_handle)

    # plot roc curves for each epoch comparing different networks
    for epoch, epoch_dict in epochs_dict.items():
        # loop over roc types
        for roc_type, net_dict in epoch_dict.items():
            if 'regr' in roc_type:
                continue
            fig_handle = plt.figure()
            # loop over networks
            for network, rates in net_dict.items():
                plt.plot(rates[1],rates[0],rates[3],label=f'ROC {network} {epoch}, auc=%0.4f'% rates[2])
            plt_fts(out_dir, roc_type, epoch, fig_handle)

        # loop over roc types
        for roc_type, net_dict in epoch_dict.items():
            # loop over different types of labels
            for i in range(4):
                if 'regr' not in roc_type:
                    continue
                fig_handle = plt.figure()
                # loop over networks
                for network, rates in net_dict.items():
                    x_t=rates[1][:, i]
                    y_r=rates[0][:, i]

                    mask= (x_t!=0.)
                    #x_t, y_r = x_t[mask], y_r[mask]

                    # plot scatter plot
                    plt.hist2d(x_t, y_r,  bins=axis_limits[i][0], cmap=plt.cm.jet, density=True)
                    plt.colorbar().set_label('Density')
                    plt_fts(out_dir, roc_type, f'scatter{axis_limits[i][2]}_{network}_{epoch}', fig_handle, axis_limits[i][1])

                fig_handle = plt.figure()
                # loop over networks
                for network, rates in net_dict.items():
                    x_t=rates[1][:, i]
                    y_r=rates[0][:, i]

                    mask= (x_t!=0.)
                    #x_t, y_r = x_t[mask], y_r[mask]

                    # plot true-reco histogram
                    plt.hist((x_t-y_r), color= rates[3],
                             bins=axis_limits[i][0][0], label=network,
                             range=(-axis_limits[i][1][1],
                             axis_limits[i][1][1]), density=True)
                    plt_fts(out_dir, roc_type, f'True-Reco{axis_limits[i][2]}_{network}_{epoch}', fig_handle, None)