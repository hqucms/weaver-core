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

plt.rcParams['agg.path.chunksize'] = 10000

'''orig_stdout = sys.stdout
f = open('roc.txt', 'w')
sys.stdout = f'''

#np.set_printoptions(threshold=np.inf)

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
# first element in the list is the label for the signal
# second element in the list is the label for the background
# third element in the list is the label
roc_type_dict={
    'aux_labels': {
        #0=b, 1=c, 2=bc, 3=other
        #'PF_SIP_b+bcVSc+other' : [[0,2], [1,3], 'pf_clas'],
        #'PF_b+bcVSc+other' : [[0,2], [1,3], 'pf_clas'],
        'PF_VtxPos' : [None, None, 'pf_regr'],
        #'PAIR_SameVtx' : [[0], None, 'pair_bin'],
    },
    # 'primary_labels':{
    #     #0=b, 1=bb, 4=uds, 5=g
    #     'JET_bVSuds':[[0,1], [4], 'primary'],
    #     'JET_bVSg':[[0,1], [5], 'primary'],
    #     'JET_bVSudsg':[[0,1], [4,5], 'primary']
    # }
}

pf_extra_fts = {
    'PF_SIP_b+bcVSc+other' : ['pf_mask_charged', 'pf_var_SIP'],
    'PF_b+bcVSc+other' : ['pf_mask_charged'],
    'PF_VtxPos' : ['pf_mask_from_b'],
}

# dictionary with the axes limits
axis_limits ={
    0: ((600, 600),[[0,2],[0,2]], 'vtx_dist_pv'),
    1: ((300, 300),[[-0.2,0.2],[-0.2,0.2]], 'vtx_x'),
    2: ((300, 300),[[-0.2,0.2],[-0.2,0.2]], 'vtx_y'),
    3: ((600, 600),[[-1, 1],[-2,2]], 'vtx_z')
}

# get the labels for the ROC curves
def get_labels(y_true, y_score, labels_s, labels_b):
    if labels_b is None:
        return y_true, y_score

    y_true_s = np.logical_or.reduce([y_true==label for label in labels_s])
    y_true_b = np.logical_or.reduce([y_true==label for label in labels_b])
    y_true_idx = np.logical_or(y_true_s,y_true_b)
    y_true_tot=y_true_s[y_true_idx].astype(int)

    if y_score.shape[1]==1:
        y_score_tot = y_score
    else:
        y_score_s=sum([y_score[:,label] for label in labels_s])*y_true_s
        y_score_b=sum([y_score[:,label] for label in labels_s])*y_true_b
        y_score_tot=y_score_s+y_score_b
        y_score_tot = y_score_tot[y_true_idx]

    return y_true_tot, y_score_tot

# compute the ROC curve and the AUC
def get_rates(y_t, y_s, l_s, l_b):
    if l_s is None and l_b is None:
        fpr, tpr, roc_auc = y_s, y_t, np.nan
    else:
        y_true, y_score = get_labels(y_t,  y_s, l_s, l_b)
        fpr, tpr, threshold = _m.roc_curve(y_true, y_score)
        roc_auc = _m.roc_auc_score(y_true, y_score)
    return fpr, tpr, roc_auc

def plt_fts(out_dir, name, fig_handle):
    if 'PF_VtxPos' in name:
        if 'True-Reco' in name:
            plt.xlabel('True-Reco')
        else:
            plt.xlabel('True')
            plt.ylabel('Reco')
            plt.plot([-40, 40], [-40, 40], 'y--', label='True = Reco')
    else:
        plt.xlabel('Efficency for b-jet (TP)')
        plt.ylabel('Mistagging prob (FP)')
        plt.ylim([1e-4, 1.005])
        plt.xlim([0.5, 1.0005])
        plt.yscale('log')

    plt.grid()
    hep.style.use('CMS')
    hep.cms.label(rlabel='')
    hep.cms.lumitext(name)
    plt.legend(labelcolor='linecolor', loc='upper left')
    #plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    fig_handle.set_size_inches(20, 15)
    plt.savefig(f'{out_dir}/{name}.png', bbox_inches='tight')
    if args.save:
        with open(f'{out_dir}/{name}.pickle', 'wb') as f:
            pickle.dump(fig_handle, f)
    if not args.not_show:
        plt.show()
    plt.close()

def load_dict(name):
    with open(name, 'r') as stream:
        loaded_dict=yaml.safe_load(stream)
    info_dict = defaultdict(list)
    for k, v in loaded_dict.items():
        # dictionary with the roc type and the epoch
        info_dict[k].append(defaultdict(defaultdict))
        info_dict[k].append(v[0])
        info_dict[k].append(v[1])
    return info_dict

def compute_roc(args, info, infile, dir_name, history, epoch):
    for label_type, labels_info in roc_type_dict.items():
        if (args.only_primary and 'primary' not in label_type) or label_type not in infile:
            continue
        print(infile)
        # load labels for each epoch and label type
        with open(os.path.join(dir_name,infile), 'rb') as f:
            file = np.load(f, allow_pickle=True)
            for roc_type, labels in labels_info.items():
                try:
                    y_true = file[f'y_true_{labels[2]}']
                except KeyError:
                    continue

                # load the extra features for the labels (if present)
                try:
                    y_score = file[pf_extra_fts[roc_type][1]]
                    print(f'EXTRA FEATURE {pf_extra_fts[roc_type][1]} found in {infile}! \n')
                except (KeyError, IndexError):
                    y_score = file[f'y_score_{labels[2]}']

                # compute roc curve for each epoch
                fpr, tpr, roc_auc=get_rates(y_true,y_score,
                                            labels[0], labels[1])

                # save roc curve for each epoch
                if args.history: info[0][roc_type][epoch]=(fpr, tpr, roc_auc)
                epochs_dict[epoch][roc_type][info[1]]=(fpr, tpr, roc_auc, info[2])

                # load the mask for the labels (if present)
                try:
                    y_mask = file[pf_extra_fts[roc_type][0]].astype(bool)
                    y_true = y_true[y_mask[:, 0]]
                    y_score = y_score[y_mask[:, 0]]
                    print(f'MASK {pf_extra_fts[roc_type][0]} found in {infile}! \n')
                    # compute roc curve for each epoch
                    fpr, tpr, roc_auc=get_rates(y_true,y_score,
                                                labels[0], labels[1])
                    # save roc curve for each epoch
                    if history: info[0][f'{roc_type}_masked'][epoch]=(fpr, tpr, roc_auc)
                    epochs_dict[epoch][f'{roc_type}_masked'][info[1]]=(fpr, tpr, roc_auc, info[2])
                except KeyError:
                    pass


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
            if epoch not in epoch_list: continue
            compute_roc(args, info, infile, dir_name, args.history, epoch)

        # compute roc curve for each epoch
        for label_type, labels_info in roc_type_dict.items():
            for roc_type, labels in labels_info.items():
                if len(info[0][roc_type]) == 0 or 'PF_VtxPos' in roc_type:
                    continue
                fig_handle = plt.figure()
                # loop over epochs
                for epoch in epoch_list:
                    #print(epoch)
                    fpr, tpr, roc_auc = info[0][roc_type][epoch]
                    plt.plot(tpr,fpr,label=f'ROC {roc_type} {info[1]} epoch #{epoch}, auc=%0.3f'% roc_auc)
                plt_fts(out_dir, f'ROC_{roc_type}_{info[1]}_history', fig_handle)

        # load files of best epoch
        for best_file in best_files:
            compute_roc(args, info, best_file, dir_name, False, 'best')

    # plot roc curves for each epoch comparing different networks
    for epoch, epoch_dict in epochs_dict.items():
        # loop over roc types
        for roc_type, networks_dict in epoch_dict.items():
            if 'PF_VtxPos' not in roc_type:
                print(roc_type)
                fig_handle = plt.figure()
                # loop over networks
                for network, rates in networks_dict.items():
                    plt.plot(rates[1],rates[0],rates[3],label=f'ROC {network} {epoch}, auc=%0.4f'% rates[2])
                plt_fts(out_dir, f"ROC_{roc_type}_{epoch}", fig_handle)

            else:
                # loop over different types of features
                for i, limits in axis_limits.items():
                    fig_handle = plt.figure()
                    # loop over networks
                    for network, rates in networks_dict.items():
                        x_t=rates[1][:, i]
                        y_r=rates[0][:, i]

                        mask= (x_t!=0.)
                        #x_t, y_r = x_t[mask], y_r[mask]

                        # plot scatter plot
                        plt.hist2d(x_t, y_r,  bins=limits[0],
                                cmap=plt.cm.jet, density=True,
                                range=limits[1])
                        plt.colorbar().set_label('Density')
                        plt_fts(out_dir,
                                f'Scatter_{roc_type}_{limits[2]}_{network}_{epoch}',
                                fig_handle)

                    fig_handle = plt.figure()
                    # loop over networks
                    for network, rates in networks_dict.items():
                        x_t=rates[1][:, i]
                        y_r=rates[0][:, i]

                        mask= (x_t!=0.)
                        #x_t, y_r = x_t[mask], y_r[mask]

                        # plot true-reco histogram
                        plt.hist((x_t-y_r), color= rates[3],
                                bins=limits[0][0], label=network,
                                range=(-limits[1][0][1],
                                limits[1][0][1]), density=True)
                        plt_fts(out_dir,
                                f'True-Reco_{roc_type}_{limits[2]}_{network}_{epoch}',
                                fig_handle)