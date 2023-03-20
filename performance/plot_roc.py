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
parser.add_argument('--in-dict', type=str, default='total',
                    help='input dictionary')
parser.add_argument('--name', type=str, default='',
                    help='name of the configuration')
parser.add_argument('--type', type=str, default='',
                    help='type of network')
args = parser.parse_args()

# type of the network
if not args.type:
    NET_TYPES = ['lite', 'full']
elif ',' in args.type:
    NET_TYPES = [k for k in args.type.split(',')]
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
    #'PF_VtxPos' : ['y_score_pf_clas'],
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
    0: ((30, 100),[[0,0.5],[0,3]], 'vtx_dist_pv', 3),
    1: ((200, 200),[[-0.06,0.06],[-0.06,0.06]], 'vtx_x', 0.2),
    2: ((200, 200),[[-0.06,0.06],[-0.06,0.06]], 'vtx_y', 0.2),
    3: ((200, 200),[[-3, 3],[-3,3]], 'vtx_z', 3),
}

def get_labels(y_true, y_score, labels_s, labels_b):
    """ Get the labels for the ROC curves
    :param    y_true : array with the true labels
    :param    y_score : array with the scores
    :param    labels_s : list with the labels for the signal
    :param    labels_b : list with the labels for the background
    :return   y_true_tot : array with the true labels for the ROC curves
    :return   y_score_tot : array with the scores for the ROC curves
    """
    if labels_b is None:
        return y_true, y_score
    # get the true label for signal and background
    y_true_s = np.logical_or.reduce([y_true==label for label in labels_s])
    y_true_b = np.logical_or.reduce([y_true==label for label in labels_b])
    # consider only the events that are signal or background
    y_true_idx = np.logical_or(y_true_s,y_true_b)
    y_true_tot=y_true_s[y_true_idx].astype(int)

    if y_score.shape[1]==1:
        y_score_tot = y_score
    else:
        # get the score for the signal and background by summing the scores
        y_score_s=sum([y_score[:,label] for label in labels_s])*y_true_s
        y_score_b=sum([y_score[:,label] for label in labels_s])*y_true_b
        y_score_tot=y_score_s+y_score_b
        # consider only the events that are signal or background
        y_score_tot = y_score_tot[y_true_idx]

    return y_true_tot, y_score_tot

def get_rates(y_t, y_s, l_s, l_b):
    """ Compute the ROC curve and the AUC
    :param    y_t : array with the true labels
    :param    y_s : array with the scores
    :param    l_s : list with the labels for the signal
    :param    l_b : list with the labels for the background
    :return   fpr : array with the false positive rate
    :return   tpr : array with the true positive rate
    :return   roc_auc : float with the AUC
    """

    if l_s is None and l_b is None:
        fpr, tpr, roc_auc = y_s, y_t, np.nan
    else:
        y_true, y_score = get_labels(y_t,  y_s, l_s, l_b)
        fpr, tpr, threshold = _m.roc_curve(y_true, y_score)
        roc_auc = _m.roc_auc_score(y_true, y_score)
    return fpr, tpr, roc_auc

def plt_fts(out_dir, name, fig_handle, AXIS_INF=None):
    """ Plot features
    :param    out_dir : string with the output directory
    :param    name : string with the name of the plot
    :param    fig_handle : figure handle
    :param    AXIS_INF : dictionary with the axes limits
    """

    if 'PF_VtxPos' in name:
        if 'True-Reco' in name:
            plt.xlabel('True-Reco [cm]', fontsize=20, loc='right')
            plt.ylabel('Density', fontsize=20, loc='top')
        else:
            plt.xlabel('True [cm]', fontsize=20, loc='right')
            plt.ylabel('Reco [cm]', fontsize=20, loc='top')
            plt.plot([-10, 10], [-10, 10], 'y--', label='True = Reco')
    else:
        plt.xlabel('Efficency for b-jet (TP)', fontsize=20, loc='right')
        plt.ylabel('Mistagging prob (FP)', fontsize=20, loc='top')
        plt.xlim([AXIS_INF[0], 1.0005])
        plt.ylim([AXIS_INF[1], 1.005])
        if 'JET' in name:
            plt.yscale('log')


    plt.grid()
    hep.style.use('CMS')
    hep.cms.label('Preliminary')
    hep.cms.label(year='UL18')
    #plt.suptitle(name, horizontalalignment='center', verticalalignment='top', fontsize=25)
    plt.legend(labelcolor='linecolor', loc='upper left')

    plt.savefig(f'{out_dir}/{name}.png', dpi = 200, bbox_inches='tight')
    if args.save:
        with open(f'{out_dir}/{name}.pickle', 'wb') as f:
            pickle.dump(fig_handle, f)
    if not args.not_show:
        plt.show()
    plt.close()

def load_dict(name):
    """ Load the dictionary from the yaml file
    :param    name : string with the name of the file
    """
    with open(name, 'r') as stream:
        loaded_dict=yaml.safe_load(stream)
    info_dict = defaultdict(list)
    for k, v in loaded_dict.items():
        # dictionary with the roc type and the epoch
        info_dict[k].append(manager.dict())
        info_dict[k].append(v[0])
        info_dict[k].append(v[1])
    return info_dict

def create_lists(input_name):
    """ Create the lists with the files and the epochs to load
    :param    input_name : string with the name of the input
    :return   files : list with the files to load
    :return   best_files : list with the best files to load
    :return   epoch_list : list with the epochs to load
    """
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

    return files, best_files, epoch_list, dir_name

def build_epochs_dict(label_dict, net_type):
    """ Build the dictionary for each epoch
    :param    label_dict : dictionary with the labels
    :param    net_type : string with the type of the network
    """
    for input_name, info in label_dict.items():
        files, best_files, epoch_list, dir_name = create_lists(input_name)
        # load files for each epoch
        for infile in files:
            epoch = int(infile.split('.npz')[0][-2:] if infile.split('.npz')[0][-2].isnumeric() else infile.split('.npz')[0][-1])
            if epoch not in epoch_list: continue
            create_dict(info, infile, dir_name, args.history, epoch, net_type)
        for best_file in best_files:
            create_dict(info, best_file, dir_name, False, 'best', net_type)


def create_dict(info, infile, dir_name, history, epoch, net_type):
    """ Create the dictionary for each epoch
    :param    info : dictionary to store the labels
    :param    infile : string with the name of the file
    :param    dir_name : string with the name of the directory
    :param    history : boolean to know if the file is a history file
    :param    epoch : epoch to load
    :param    net_type : string with the type of the network
    """
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

                    if history: info[0][f'{roc_type}_mask']= manager.dict()
                    EPOCHS_DICT[net_type][epoch][f'{roc_type}_mask'] = manager.dict()


def compute_roc(info, infile, dir_name, history, epoch, net_type):
    """ Compute the roc for each epoch
    :param    info : dictionary to store the labels
    :param    infile : string with the name of the file
    :param    dir_name : string with the name of the directory
    :param    history : boolean to know if the file is a history file
    :param    epoch : epoch to load
    :param    net_type : string with the type of the network
    """
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
                    #print(f'EXTRA FEATURE {PF_EXTRA_FTS[roc_type][1]} found in {infile}! \n')
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
                    if PF_EXTRA_FTS[roc_type][0] == 'y_score_pf_clas':
                        y_mask = np.expand_dims(np.argmax(y_mask, axis=1), axis=1) == 0
                    y_true = y_true[y_mask[:, 0]]
                    y_score = y_score[y_mask[:, 0]]
                    #print(f'MASK {PF_EXTRA_FTS[roc_type][0]} found in {infile}! \n')
                    # compute roc curve for each epoch
                    fpr, tpr, roc_auc=get_rates(y_true,y_score,
                                                labels[0], labels[1])
                    # save roc curve for each epoch
                    if history: info[0][f'{roc_type}_mask'][epoch]=(fpr, tpr, roc_auc)
                    EPOCHS_DICT[net_type][epoch][f'{roc_type}_mask'][info[1]]=(fpr, tpr, roc_auc, info[2])
                except KeyError:
                    pass

def plotting_history_function(epoch_list, info,  roc_type, out_dir, net_type):
    """ Plot the roc curves for each epoch
    :param    epoch_list : list of epochs to plot
    :param    info : dictionary with the labels
    :param    roc_type : string with the type of the roc
    :param    out_dir : string with the name of the output directory
    :param    net_type : string with the type of the network
    """
    fig_handle = plt.figure(figsize=(20, 15))
    # loop over epochs
    for epoch in epoch_list:
        fpr, tpr, roc_auc = info[0][roc_type][epoch]
        plt.plot(tpr,fpr,label=f'ROC {roc_type} {info[1]} epoch #{epoch}, auc=%0.3f'% roc_auc)
    plt_fts(out_dir, f'ROC_{roc_type}_{info[1]}_{net_type}_history', fig_handle)

def plotting_function(out_dir, epoch, roc_type, networks, net_type, networks_2 = None, name1 = '', name2 = '', network_name=''):
    """ Plot the roc curves for a epoch and a roc type for each network
    :param    out_dir : string with the name of the output directory
    :param    epoch : epoch to plot
    :param    roc_type : string with the type of the roc
    :param    networks :  networks to plot
    :param    net_type : string with the type of the network
    :param    networks_2 : other networks to plot
    :param    name1 : string with the name of the first network
    :param    name2 : string with the name of the second network
    :param    network_name : string with the name of the network
    """

    if 'PF_VtxPos' not in roc_type:
        fig_handle = plt.figure(figsize=(20, 15))
        # loop over networks
        if isinstance(networks, mp.managers.DictProxy):
            for network, rates in networks.items():
                plt.plot(rates[1],rates[0],rates[3],label=f'ROC {network} {epoch}, auc=%0.4f'% rates[2])
        elif isinstance(networks, tuple):
            rates=networks
            plt.plot(rates[1],rates[0],rates[3],label=f'ROC {network_name}{name1} {epoch}, auc=%0.4f'% rates[2])

        if isinstance(networks_2, mp.managers.DictProxy):
            for network, rates in networks_2.items():
                plt.plot(rates[1],rates[0],color=f'{rates[3]}', linestyle='dotted', label=f'ROC {network}{name2} {epoch}, auc=%0.4f'% rates[2])
        elif isinstance(networks_2, tuple):
            rates=networks_2
            plt.plot(rates[1],rates[0],color=f'{rates[3]}', linestyle='dotted', label=f'ROC {network_name}{name2} {epoch}, auc=%0.4f'% rates[2])

        plt_fts(out_dir, f"ROC_{roc_type}_{net_type}_{network_name}{epoch}", fig_handle, AXIS_INF[roc_type.replace('_mask', '')])

    else:
        out_dir_scatter = os.path.join(out_dir, 'Scatter')
        os.makedirs(out_dir_scatter, exist_ok=True)

        out_dir_true_reco = os.path.join(out_dir, 'True_Reco')
        os.makedirs(out_dir_true_reco, exist_ok=True)

        # loop over different types of features
        for i, limits in AXIS_LIMITS.items():
            # loop over networks
            for network, rates in networks.items():
                x_t=rates[1][:, i]
                y_r=rates[0][:, i]

                for mask_name, mask in {'_notZero': x_t != 0, '': np.ones_like(x_t, dtype=bool)}.items():
                    fig_handle = plt.figure(figsize=(20, 15))
                    ax = plt.gca()
                    x_t_mask, y_r_mask = x_t[mask], y_r[mask]

                    # plot scatter plot
                    plt.hist2d(x_t_mask, y_r_mask,  bins=limits[0],
                            cmap=plt.cm.jet, density=True,
                            range=limits[1])

                    cmap = mpl.cm.jet
                    norm = mpl.colors.Normalize(vmin=0, vmax=1.0)
                    plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax).set_label('Normalized counts', loc='center', fontsize=20)
                    plt_fts(out_dir_scatter,
                            f'Scatter_{roc_type}{mask_name}_{limits[2]}_{network}_{net_type}_{epoch}',
                            fig_handle)

            # loop over networks
            for network, rates in networks.items():
                x_t=rates[1][:, i]
                y_r=rates[0][:, i]

                for mask_name, mask in {'_notZero': x_t != 0, '': np.ones_like(x_t, dtype=bool)}.items():
                    fig_handle = plt.figure(figsize=(20, 15))
                    x_t_mask, y_r_mask = x_t[mask], y_r[mask]

                    # plot true-reco histogram
                    plt.hist((x_t_mask-y_r_mask), color= rates[3],
                            bins=limits[0][0], label=network,
                            range=(-limits[3],
                            limits[3]), density=True)
                    plt_fts(out_dir_true_reco,
                            f'True-Reco_{roc_type}{mask_name}_{limits[2]}_{network}_{net_type}_{epoch}',
                            fig_handle)
        networks.clear()


def _main(net_type, out_dir, label_dict):
    """ Main function
    :param    net_type : string with the type of the network
    :param    out_dir : string with the name of the output directory
    :param    label_dict : dictionary with the labels
    """

    # create output directory
    out_dir = os.path.join(out_dir, f'{args.name}_roc_{net_type}')
    os.makedirs(out_dir, exist_ok=True)

    build_epochs_dict(label_dict, net_type)
    print('\n Finished building epochs dict \n')

    parallel_list=[]
    for input_name, info in label_dict.items():
        files, best_files, epoch_list, dir_name = create_lists(input_name)


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

        for mask in ['', '_mask']:
            if f'PF_SIP_b+bcVSc+other{mask}' in epoch_dict.keys() and f'PF_b+bcVSc+other{mask}' in epoch_dict.keys():
                p=mp.Process(target=plotting_function,
                                    args=(out_dir, epoch, f'PF_comparison_b+bcVSc+other{mask}',
                                            epoch_dict[f'PF_b+bcVSc+other{mask}'], net_type,
                                            epoch_dict[f'PF_SIP_b+bcVSc+other{mask}'], '',  '_SIP'))
                p.start()
                parallel_list.append(p)


    # Join parallel
    for parallel_elem in parallel_list:
        parallel_elem.join()


if __name__ == '__main__':
    start=time.time()

    date_time = time.strftime('%Y%m%d-%H%M%S')
    out_dir = os.path.join(f'{args.out_path}roc_curve', f'{date_time}_{args.name}_{args.in_dict}_roc')
    print(f'Output directory: {out_dir}')

    parallel_list=[]
    for net_type in NET_TYPES:
        label_dict=load_dict(f'{args.in_dict}_{net_type}.yaml')
        p=mp.Process(target=_main,
                        args=(net_type, out_dir, label_dict))
        p.start()
        parallel_list.append(p)
    # Join parallel
    for parallel_elem in parallel_list:
        parallel_elem.join()

    print('starting comparison')
    parallel_list=[]

    if len(NET_TYPES) > 1:
        label_dict=load_dict(f'{args.in_dict}_{NET_TYPES[0]}.yaml')
        input_name = list(label_dict.keys())[0]
        _, _, epoch_list, _ = create_lists(input_name)
        epoch_list.append('best')

        os.makedirs(f'{out_dir}/net_type_comparison', exist_ok=True)

        for info in label_dict.values():
            network=info[1]
            #print(network)
            for label_type, labels_info in ROC_TYPE_DICT.items():
                for roc_type in labels_info.keys():
                    if 'PF_VtxPos' == roc_type:
                        continue
                    for epoch in epoch_list:
                        for mask in ['', '_mask']:
                            #print(roc_type)
                            try:
                                p=mp.Process(target=plotting_function,
                                                    args=(f'{out_dir}/net_type_comparison', epoch, f'{roc_type}{mask}',
                                                    EPOCHS_DICT[NET_TYPES[0]][epoch][f'{roc_type}{mask}'][network], 'net_type_comparison',
                                                    EPOCHS_DICT[NET_TYPES[1]][epoch][f'{roc_type}{mask}'][network],
                                                    NET_TYPES[0], NET_TYPES[1], f'{network}_'))
                                p.start()
                                parallel_list.append(p)
                            except KeyError:
                                pass
        # Join parallel
        for parallel_elem in parallel_list:
            parallel_elem.join()

    print('Total time:   ', time.time()-start)
