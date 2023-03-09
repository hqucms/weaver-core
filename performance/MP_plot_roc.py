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
import multiprocessing as mp

manager = mp.Manager()

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
parser.add_argument('--in-path', type=str, default='',
                    help='input path')
parser.add_argument('--out-path', type=str, default='',
                    help='output path')
parser.add_argument('--name', type=str, default='',
                    help='name of the configuration')
parser.add_argument('--in-dict', type=str, default='performance_comparison',
                    help='name of the file with the dictionary')
args = parser.parse_args()

# dictionary where the first key is the epoch,
# the second is the type of ROC
# and the third is the network
epochs_dict=manager.dict()
#epochs_dict=defaultdict(lambda: defaultdict(defaultdict))

# dictionary with the labels for the ROC curves
# first element in the list is the label for the signal
# second element in the list is the label for the background
# third element in the list is the label
roc_type_dict={
    'aux_labels': {
        #0=b, 1=c, 2=bc, 3=other
        'PF_SIP_b+bcVSc+other' : [[0,2], [1,3], 'pf_clas'],
        'PF_b+bcVSc+other' : [[0,2], [1,3], 'pf_clas'],
        'PF_VtxPos' : [None, None, 'pf_regr'],
        'PAIR_SameVtx' : [[0], None, 'pair_bin'],
    },
    'primary_labels':{
        #0=b, 1=bb, 4=uds, 5=g
        'JET_bVSuds':[[0,1], [4], 'primary'],
        'JET_bVSg':[[0,1], [5], 'primary'],
        'JET_bVSudsg':[[0,1], [4,5], 'primary']
    }
}

pf_extra_fts = {
    'PF_SIP_b+bcVSc+other' : ['pf_mask_charged', 'pf_var_IPsig'],
    'PF_b+bcVSc+other' : ['pf_mask_charged'],
    'PF_VtxPos' : ['pf_mask_from_b'],
}

# dictionary with the axes inf
axis_inf ={
    'PF_SIP_b+bcVSc+other': (0.4, 5e-3),
    'PF_b+bcVSc+other':  (0.4, 5e-3),
    'PAIR_SameVtx':  (0.4, 1e-1),
    'JET_bVSuds':  (0.5, 1e-4),
    'JET_bVSg': (0.5, 1e-4),
    'JET_bVSudsg': (0.5, 1e-4),
}

# dictionary with the axes limits
axis_limits ={
    0: ((100, 100),[[0,3],[0,3]], 'vtx_dist_pv'),
    1: ((100, 100),[[-0.05,0.05],[-0.05,0.05]], 'vtx_x'),
    2: ((100, 100),[[-0.05,0.05],[-0.05,0.05]], 'vtx_y'),
    3: ((100, 100),[[-2, 2],[-2,2]], 'vtx_z'),
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

def plt_fts(out_dir, name, fig_handle, axis_inf=None):
    if 'PF_VtxPos' in name:
        if 'True-Reco' in name:
            plt.xlabel('True-Reco [cm]')
        else:
            plt.xlabel('True [cm]')
            plt.ylabel('Reco [cm]')
            plt.plot([-10, 10], [-10, 10], 'y--', label='True = Reco')
    else:
        plt.xlabel('Efficency for b-jet (TP)')
        plt.ylabel('Mistagging prob (FP)')
        plt.xlim([axis_inf[0], 1.0005])
        plt.ylim([axis_inf[1], 1.005])
        plt.yscale('log')

    plt.grid()
    hep.style.use('CMS')
    hep.cms.label(rlabel='')
    hep.cms.lumitext(name)
    #fig_handle.set_size_inches(20, 15))
    plt.legend(labelcolor='linecolor', loc='upper left')

    plt.savefig(f'{out_dir}/{name}.png', dpi = 200, bbox_inches='tight')
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
        info_dict[k].append(manager.dict())
        info_dict[k].append(v[0])
        info_dict[k].append(v[1])
    return info_dict

def build_epochs_dict():

    for input_name, info in label_dict.items():
        # files to load
        dir_name=f'{args.in_path}{input_name}'
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
            create_dict(info, infile, dir_name, args.history, epoch)
        for best_file in best_files:
            create_dict(info, best_file, dir_name, False, 'best')


def create_dict(info, infile, dir_name, history, epoch):
    if epoch not in epochs_dict.keys():
        epochs_dict[epoch] = manager.dict()
    for label_type, labels_info in roc_type_dict.items():
        #print(label_type)
        if (args.only_primary and 'primary' not in label_type) or label_type not in infile:
            continue
        #print(infile)
        # load labels for each epoch and label type
        with open(os.path.join(dir_name,infile), 'rb') as f:
            file = np.load(f, allow_pickle=True)
            for roc_type, labels in labels_info.items():
                if roc_type in epochs_dict[epoch].keys(): continue
                # load the extra features for the labels (if present)
                try:
                    _ = file[pf_extra_fts[roc_type][1]]
                    print(f'EXTRA FEATURE {pf_extra_fts[roc_type][1]} found in {infile}! \n')
                except (KeyError, IndexError):
                    try:
                        _ = file[f'y_score_{labels[2]}']
                    except KeyError:
                        continue

                # save roc curve for each epoch
                if history: info[0][roc_type] = manager.dict()
                epochs_dict[epoch][roc_type] = manager.dict()

                # load the mask for the labels (if present)
                try:
                    _ = file[pf_extra_fts[roc_type][0]].astype(bool)
                    print(f'MASK {pf_extra_fts[roc_type][0]} found in {infile}! \n')
                    # save roc curve for each epoch
                    if history: info[0][f'{roc_type}_masked']= manager.dict()
                    epochs_dict[epoch][f'{roc_type}_masked'] = manager.dict()
                except KeyError:
                    pass

def compute_roc(info, infile, dir_name, history, epoch):
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
                    try:
                        y_score = file[f'y_score_{labels[2]}']
                    except KeyError:
                        continue

                # compute roc curve for each epoch
                fpr, tpr, roc_auc=get_rates(y_true,y_score,
                                            labels[0], labels[1])

                # save roc curve for each epoch
                if history: info[0][roc_type][epoch]=(fpr, tpr, roc_auc)
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

def plotting_history_function(epoch_list, info,  roc_type, out_dir):
    fig_handle = plt.figure(figsize=(20, 15))
    # loop over epochs
    for epoch in epoch_list:
        fpr, tpr, roc_auc = info[0][roc_type][epoch]
        plt.plot(tpr,fpr,label=f'ROC {roc_type} {info[1]} epoch #{epoch}, auc=%0.3f'% roc_auc)
    plt_fts(out_dir, f'ROC_{roc_type}_{info[1]}_history', fig_handle)

def plotting_function(epoch, roc_type, networks_dict, networks_dict_ = None):
    if 'PF_VtxPos' not in roc_type:
        fig_handle = plt.figure(figsize=(20, 15))
        # loop over networks
        for network, rates in networks_dict.items():
            plt.plot(rates[1],rates[0],rates[3],label=f'ROC {network} {epoch}, auc=%0.4f'% rates[2])
        if networks_dict_ is not None:
            for network, rates in networks_dict_.items():
                plt.plot(rates[1],rates[0],f'{rates[3]}--',label=f'ROC {network} {epoch}, auc=%0.4f'% rates[2])
        plt_fts(out_dir, f"ROC_{roc_type}_{epoch}", fig_handle, axis_inf[roc_type])
    else:
        # loop over different types of features
        for i, limits in axis_limits.items():
            fig_handle = plt.figure(figsize=(20, 15))
            # loop over networks
            for network, rates in networks_dict.items():
                x_t=rates[1][:, i]
                y_r=rates[0][:, i]

                # plot scatter plot
                plt.hist2d(x_t, y_r,  bins=limits[0],
                        cmap=plt.cm.jet, density=True,
                        range=limits[1])
                plt.colorbar().set_label('Density')
                plt_fts(out_dir,
                        f'Scatter_{roc_type}_{limits[2]}_{network}_{epoch}',
                        fig_handle)

            fig_handle = plt.figure(figsize=(20, 15))
            # loop over networks
            for network, rates in networks_dict.items():
                x_t=rates[1][:, i]
                y_r=rates[0][:, i]

                # plot true-reco histogram
                plt.hist((x_t-y_r), color= rates[3],
                        bins=limits[0][0], label=network,
                        range=(-limits[1][0][1],
                        limits[1][0][1]), density=True)
                plt_fts(out_dir,
                        f'True-Reco_{roc_type}_{limits[2]}_{network}_{epoch}',
                        fig_handle)


if __name__ == '__main__':
    start=time.time()
    print('###################################################################################################################################')

    label_dict=load_dict(f'{args.in_dict}.yaml')

    # create output directory
    date_time = time.strftime('%Y%m%d-%H%M%S')
    out_dir = os.path.join(f'{args.out_path}roc_curve', f'{date_time}_{args.name}_roc')
    os.makedirs(out_dir, exist_ok=True)

    build_epochs_dict()
    print('\n done building epochs dict \n')

    parallel_list=[]
    for input_name, info in label_dict.items():
        # files to load
        dir_name=f'{args.in_path}{input_name}'
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
            p=mp.Process(target=compute_roc,
                                args=(info, infile, dir_name, args.history, epoch))
            p.start()
            parallel_list.append(p)
            #compute_roc(info, infile, dir_name, args.history, epoch)

        # load files of best epoch
        for best_file in best_files:
            p=mp.Process(target=compute_roc,
                                args=(info, best_file, dir_name, False, 'best'))
            p.start()
            parallel_list.append(p)
            #compute_roc(info, best_file, dir_name, False, 'best')

    # Join parallel
    for parallel_elem in parallel_list:
        parallel_elem.join()

    # print(epochs_dict)
    # print(epochs_dict['best'])
    # print(epochs_dict['best']['PF_SIP_b+bcVSc+other']['clas'][0][:100])
    # print(epochs_dict['best']['PF_b+bcVSc+other']['clas'][0][:100])

    print('\n start plotting \n')
    parallel_list = []
    for input_name, info in label_dict.items():
        # compute roc curve for each epoch
        for label_type, labels_info in roc_type_dict.items():
            for roc_type, labels in labels_info.items():
                if roc_type not in info[0].keys() or 'PF_VtxPos' == roc_type:
                    continue
                p=mp.Process(target=plotting_history_function,
                                    args=(epoch_list, info,  roc_type, out_dir))
                p.start()
                parallel_list.append(p)

    # Join parallel
    for parallel_elem in parallel_list:
        parallel_elem.join()



    parallel_list = []
    # plot roc curves for each epoch comparing different networks
    for epoch, epoch_dict in epochs_dict.items():
        # loop over roc types
        for roc_type, networks_dict in epoch_dict.items():
            p=mp.Process(target=plotting_function,
                                args=(epoch, roc_type, networks_dict))
            p.start()
            parallel_list.append(p)
        if 'PF_SIP_b+bcVSc+other' and 'PF_b+bcVSc+other' in epoch_dict.keys():
            p=mp.Process(target=plotting_function,
                                args=(epoch, 'PF_comparison_b+bcVSc+other', epoch_dict['PF_b+bcVSc+other'], epoch_dict['PF_SIP_b+bcVSc+other']))
            p.start()
            parallel_list.append(p)


    # Join parallel
    for parallel_elem in parallel_list:
        parallel_elem.join()

    print('Total time:   ', time.time()-start)
