import numpy as np
import sklearn.metrics as _m
import matplotlib.pyplot as plt
import matplotlib as mpl
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
parser.add_argument('--type', type=str, default='',
                    help='name of the file with the dictionary')
args = parser.parse_args()

if not args.type:
    NET_TYPES = ['lite', 'full']
else:
    NET_TYPES = [args.type]

# dictionary where the first key is the net type (lite, full, etc.)
# the second is the epoch,
# the third is the type of ROC
# and the fourth is the network
EPOCHS_DICT = manager.dict()
for _net_type in NET_TYPES:
    EPOCHS_DICT[_net_type] = manager.dict()


# dictionary with the labels for the ROC curves
# first element in the list is the label for the signal
# second element in the list is the label for the background
# third element in the list is the label
ROC_TYPE_DICT={
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

PF_EXTRA_FTS = {
    'PF_SIP_b+bcVSc+other' : ['pf_mask_charged', 'pf_var_IPsig'],
    'PF_b+bcVSc+other' : ['pf_mask_charged'],
    'PF_VtxPos' : ['pf_mask_from_b'],
}

# dictionary with the axes inf
AXIS_INF ={
    'PF_SIP_b+bcVSc+other': (0.4, 5e-3),
    'PF_b+bcVSc+other':  (0.4, 5e-3),
    'PF_comparison_b+bcVSc+other':  (0.4, 5e-3),
    'PAIR_SameVtx':  (0.4, 1e-1),
    'JET_bVSuds':  (0.5, 1e-4),
    'JET_bVSg': (0.5, 1e-4),
    'JET_bVSudsg': (0.5, 1e-4),
}

# dictionary with the axes limits
AXIS_LIMITS ={
    0: ((100, 100),[[0,3],[0,3]], 'vtx_dist_pv', 3),
    1: ((100, 100),[[-0.2,0.2],[-0.2,0.2]], 'vtx_x', 0.2),
    2: ((100, 100),[[-0.2,0.2],[-0.2,0.2]], 'vtx_y', 0.2),
    3: ((100, 100),[[-2, 2],[-2,2]], 'vtx_z', 3),
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

def plt_fts(out_dir, name, fig_handle, AXIS_INF=None):
    if 'PF_VtxPos' in name:
        if 'True-Reco' in name:
            plt.xlabel('True-Reco [cm]')
            plt.ylabel('Density')
        else:
            plt.xlabel('True [cm]')
            plt.ylabel('Reco [cm]')
            plt.plot([-10, 10], [-10, 10], 'y--', label='True = Reco')
    else:
        plt.xlabel('Efficency for b-jet (TP)')
        plt.ylabel('Mistagging prob (FP)')
        plt.xlim([AXIS_INF[0], 1.0005])
        plt.ylim([AXIS_INF[1], 1.005])
        plt.yscale('log')
        #hep.cms.lumitext(name)


    plt.grid()
    hep.style.use('CMS')
    hep.cms.label(rlabel='')
    plt.suptitle(name, horizontalalignment='center', verticalalignment='top', fontsize=25)
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

def build_epochs_dict(label_dict, net_type):
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
            create_dict(info, infile, dir_name, args.history, epoch, net_type)
        for best_file in best_files:
            create_dict(info, best_file, dir_name, False, 'best', net_type)


def create_dict(info, infile, dir_name, history, epoch, net_type):
    if epoch not in EPOCHS_DICT[net_type].keys():
        EPOCHS_DICT[net_type][epoch] = manager.dict()
    for label_type, labels_info in ROC_TYPE_DICT.items():
        #print(label_type)
        if (args.only_primary and 'primary' not in label_type) or label_type not in infile:
            continue
        #print(infile)
        # load labels for each epoch and label type
        with open(os.path.join(dir_name,infile), 'rb') as f:
            file = np.load(f, allow_pickle=True, mmap_mode='r')
            for roc_type, labels in labels_info.items():
                if roc_type in EPOCHS_DICT[net_type][epoch].keys(): continue

                # load the extra features for the labels (if present)
                if (roc_type in PF_EXTRA_FTS.keys() \
                    and ((len(PF_EXTRA_FTS[roc_type])==2 \
                    and PF_EXTRA_FTS[roc_type][1] in file.files) \
                    or (len(PF_EXTRA_FTS[roc_type])==1 \
                    and f'y_score_{labels[2]}' in file.files)))\
                    or (roc_type not in PF_EXTRA_FTS.keys() \
                    and f'y_score_{labels[2]}' in file.files):

                    if history: info[0][roc_type] = manager.dict()
                    EPOCHS_DICT[net_type][epoch][roc_type] = manager.dict()

                if roc_type in PF_EXTRA_FTS.keys() \
                    and PF_EXTRA_FTS[roc_type][0] in file.files:

                    if history: info[0][f'{roc_type}_masked']= manager.dict()
                    EPOCHS_DICT[net_type][epoch][f'{roc_type}_masked'] = manager.dict()


def compute_roc(info, infile, dir_name, history, epoch, net_type):
    for label_type, labels_info in ROC_TYPE_DICT.items():
        if (args.only_primary and 'primary' not in label_type) or label_type not in infile:
            continue
        print(infile)
        # load labels for each epoch and label type
        with open(os.path.join(dir_name,infile), 'rb') as f:
            file = np.load(f, allow_pickle=True, mmap_mode='r')
            for roc_type, labels in labels_info.items():
                try:
                    y_true = file[f'y_true_{labels[2]}']
                except KeyError:
                    continue

                # load the extra features for the labels (if present)
                try:
                    y_score = file[PF_EXTRA_FTS[roc_type][1]]
                    print(f'EXTRA FEATURE {PF_EXTRA_FTS[roc_type][1]} found in {infile}! \n')
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
                EPOCHS_DICT[net_type][epoch][roc_type][info[1]]=(fpr, tpr, roc_auc, info[2])

                # load the mask for the labels (if present)
                try:
                    y_mask = file[PF_EXTRA_FTS[roc_type][0]].astype(bool)
                    y_true = y_true[y_mask[:, 0]]
                    y_score = y_score[y_mask[:, 0]]
                    print(f'MASK {PF_EXTRA_FTS[roc_type][0]} found in {infile}! \n')
                    # compute roc curve for each epoch
                    fpr, tpr, roc_auc=get_rates(y_true,y_score,
                                                labels[0], labels[1])
                    # save roc curve for each epoch
                    if history: info[0][f'{roc_type}_masked'][epoch]=(fpr, tpr, roc_auc)
                    EPOCHS_DICT[net_type][epoch][f'{roc_type}_masked'][info[1]]=(fpr, tpr, roc_auc, info[2])
                except KeyError:
                    pass

def plotting_history_function(epoch_list, info,  roc_type, out_dir, net_type):
    fig_handle = plt.figure(figsize=(20, 15))
    # loop over epochs
    for epoch in epoch_list:
        fpr, tpr, roc_auc = info[0][roc_type][epoch]
        plt.plot(tpr,fpr,label=f'ROC {roc_type} {info[1]} epoch #{epoch}, auc=%0.3f'% roc_auc)
    plt_fts(out_dir, f'ROC_{roc_type}_{info[1]}_{net_type}_history', fig_handle)

def plotting_function(out_dir, epoch, roc_type, networks_dict, net_type, networks_dict_ = None, name1 = '', name2 = '', network_name=''):
    if 'PF_VtxPos' not in roc_type:
        fig_handle = plt.figure(figsize=(20, 15))
        # loop over networks
        if isinstance(networks_dict, dict):
            for network, rates in networks_dict.items():
                plt.plot(rates[1],rates[0],rates[3],label=f'ROC {network} {epoch}, auc=%0.4f'% rates[2])
        else:
            rates=networks_dict
            plt.plot(rates[1],rates[0],rates[3],label=f'ROC {network_name}{name1} {epoch}, auc=%0.4f'% rates[2])

        if networks_dict_ is not None:
            for network, rates in networks_dict_.items():
                plt.plot(rates[1],rates[0],color=f'{rates[3]}', linestyle='dashed', label=f'ROC {network}{name2} {epoch}, auc=%0.4f'% rates[2])
        plt_fts(out_dir, f"ROC_{roc_type}_{net_type}_{epoch}", fig_handle, AXIS_INF[roc_type.replace('_masked', '')])
    else:
        # loop over different types of features
        for i, limits in AXIS_LIMITS.items():
            fig_handle = plt.figure(figsize=(20, 15))
            # loop over networks
            for network, rates in networks_dict.items():
                x_t=rates[1][:, i]
                y_r=rates[0][:, i]

                # plot scatter plot
                plt.hist2d(x_t, y_r,  bins=limits[0],
                        cmap=plt.cm.jet, density=True,
                        range=limits[1])

                cmap = mpl.cm.jet
                norm = mpl.colors.Normalize(vmin=0, vmax=1.0)
                plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap)).set_label('Normalized counts', loc='center', fontsize=20)

                # histo, xedges, yedges = np.histogram2d(x_t, y_r, bins=limits[0], density=True, range=limits[1])
                # histo_normalized = histo/histo.max((0,1))
                # extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
                # im = axis.imshow(histo_normalized, cmap=plt.cm.jet, interpolation='none',origin ='lower')
                #plt.colorbar(im, ax=axis).set_label('Density', loc='center', fontsize=20)

                plt_fts(out_dir,
                        f'Scatter_{roc_type}_{limits[2]}_{network}_{net_type}_{epoch}',
                        fig_handle)

            fig_handle = plt.figure(figsize=(20, 15))
            # loop over networks
            for network, rates in networks_dict.items():
                x_t=rates[1][:, i]
                y_r=rates[0][:, i]

                # plot true-reco histogram
                plt.hist((x_t-y_r), color= rates[3],
                        bins=limits[0][0], label=network,
                        range=(-limits[3],
                        limits[3]), density=True)
                plt_fts(out_dir,
                        f'True-Reco_{roc_type}_{limits[2]}_{network}_{net_type}_{epoch}',
                        fig_handle)
        networks_dict.clear()


def _main(net_type, out_dir, label_dict):

    # create output directory
    out_dir = os.path.join(out_dir, f'{args.name}_roc_{net_type}')
    os.makedirs(out_dir, exist_ok=True)

    build_epochs_dict(label_dict, net_type)
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
                                args=(info, infile, dir_name, args.history, epoch, net_type))
            p.start()
            parallel_list.append(p)

        # load files of best epoch
        for best_file in best_files:
            p=mp.Process(target=compute_roc,
                                args=(info, best_file, dir_name, False, 'best', net_type))
            p.start()
            parallel_list.append(p)

    # Join parallel
    for parallel_elem in parallel_list:
        parallel_elem.join()

    print('\n start plotting \n')
    parallel_list = []
    for input_name, info in label_dict.items():
        # compute roc curve for each epoch
        for label_type, labels_info in ROC_TYPE_DICT.items():
            for roc_type, labels in labels_info.items():
                if roc_type not in info[0].keys() or 'PF_VtxPos' == roc_type:
                    continue
                p=mp.Process(target=plotting_history_function,
                                    args=(epoch_list, info,  roc_type, out_dir, net_type))
                p.start()
                parallel_list.append(p)

    # Join parallel
    for parallel_elem in parallel_list:
        parallel_elem.join()


    parallel_list = []
    # plot roc curves for each epoch comparing different networks
    for epoch, epoch_dict in EPOCHS_DICT[net_type].items():
        # loop over roc types
        for roc_type, networks_dict in epoch_dict.items():
            p=mp.Process(target=plotting_function,
                                args=(out_dir, epoch, roc_type, networks_dict, net_type))
            p.start()
            parallel_list.append(p)

        for mask in ['', '_masked']:
            if f'PF_SIP_b+bcVSc+other{mask}' in epoch_dict.keys() and f'PF_b+bcVSc+other{mask}' in epoch_dict.keys():
                p=mp.Process(target=plotting_function,
                                    args=(out_dir, epoch, f'PF_comparison_b+bcVSc+other{mask}',
                                            epoch_dict[f'PF_b+bcVSc+other{mask}'], net_type,
                                            epoch_dict[f'PF_SIP_b+bcVSc+other{mask}'],'',  '_SIP'))
                p.start()
                parallel_list.append(p)


    # Join parallel
    for parallel_elem in parallel_list:
        parallel_elem.join()


if __name__ == '__main__':
    start=time.time()

    print('###################################################################################################################################')

    date_time = time.strftime('%Y%m%d-%H%M%S')
    out_dir = os.path.join(f'{args.out_path}roc_curve', f'{date_time}_{args.name}_roc')

    parallel_list=[]
    for net_type in NET_TYPES:
        label_dict=load_dict(f'performance_comparison_{net_type}.yaml')
        p=mp.Process(target=_main,
                        args=(net_type, out_dir, label_dict))
        p.start()
        parallel_list.append(p)
    # Join parallel
    for parallel_elem in parallel_list:
        parallel_elem.join()

    print('starting comparison')
    parallel_list=[]
    for info in label_dict.values():
        network=info[1]
        print(network)
        for label_type, labels_info in ROC_TYPE_DICT.items():
            for roc_type in labels_info.keys():
                if 'PF_VtxPos' == roc_type:
                    continue
                print(roc_type)
                try:
                    p=mp.Process(target=plotting_function,
                                        args=(out_dir, 'best', roc_type,
                                        EPOCHS_DICT[NET_TYPES[0]]['best'][roc_type][network], 'comparison',
                                        EPOCHS_DICT[NET_TYPES[1]]['best'][roc_type][network],
                                        NET_TYPES[0], NET_TYPES[1], network))
                    p.start()
                    parallel_list.append(p)
                except KeyError:
                    pass
    # Join parallel
    for parallel_elem in parallel_list:
        parallel_elem.join()

    print('Total time:   ', time.time()-start)
