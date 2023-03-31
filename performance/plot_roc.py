import numpy as np
import sklearn.metrics as _m
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
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
parser.add_argument('--complete-dict', type=str, default='complete_dict',
                    help='dictionary with names')
parser.add_argument('--name', type=str, default='',
                    help='name of the configuration')
parser.add_argument('--type', type=str, default='lite,full',
                    help='type of network')
parser.add_argument('--roc-config', type=str, default='roc_config',
                    help='name of the file with the dictionary')
args = parser.parse_args()

# type of the network
if ',' in args.type:
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


with open(f'{args.roc_config}.yaml', 'r') as stream:
    config_dicts=yaml.safe_load(stream)

ROC_TYPE_DICT=config_dicts['ROC_TYPE_DICT']
PF_EXTRA_FTS=config_dicts['PF_EXTRA_FTS']
AXIS_INF=config_dicts['AXIS_INF']
AXIS_LIMITS=config_dicts['AXIS_LIMITS']
SPECIAL_DICT=config_dicts['SPECIAL_DICT']

def find_matching_suffix_pairs(d):
    matching_pairs = []
    keys = list(d.keys())
    for i in range(len(keys)):
        for j in range(i+1, len(keys)):
            prefix_i = keys[i].replace('_mask', '')
            prefix_j = keys[j].replace('_mask', '')
            if ('_mask' in keys[i] and '_mask' not in keys[j]) or \
                ('_mask' not in keys[i] and '_mask' in keys[j]):# or \
            #    prefix_i == prefix_j:
                continue
            suffix_i = prefix_i.split('_')[-1]
            suffix_j = prefix_j.split('_')[-1]
            if suffix_i == suffix_j:
                matching_pairs.append((keys[i], keys[j], suffix_i))
    return matching_pairs

def find_matching_suffix_groups(d):
    groups = {k.replace('_mask', '').rsplit('_', 1)[1]: [] for k in d}
    for k in d:
        groups[k.replace('_mask', '').rsplit('_', 1)[1]].append(k)
    sorted_groups = {suffix: sorted(keys, key=lambda k: get_middle_substring(k.replace('_mask', '')) != '') for suffix, keys in groups.items()}
    return {suffix: keys for suffix, keys in sorted_groups.items() if len(keys) > 1}

def get_middle_substring(s):
    substrings = s.split('_')
    if len(substrings) == 3:
        return substrings[1]
    else:
        return ""


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

def plt_fts(out_dir, name, fig_handle, axis_inf=None):
    """ Plot features
    :param    out_dir : string with the output directory
    :param    name : string with the name of the plot
    :param    fig_handle : figure handle
    :param    axis_inf : dictionary with the axes limits
    """

    if SPECIAL_DICT['Scatter_True-Reco'] in name:
        if 'True-Reco' in name:
            plt.xlabel('True-Reco [cm]', fontsize=20, loc='right')
            plt.ylabel('Density', fontsize=20, loc='top')
        else:
            plt.xlabel('True [cm]', fontsize=20, loc='right')
            plt.ylabel('Reco [cm]', fontsize=20, loc='top')
            plt.plot([-10, 10], [-10, 10], 'y--', label='True = Reco')
    else:
        plt.xlabel('b-jet efficiency (TP)', fontsize=20, loc='right')
        plt.ylabel('Mistagging probability (FP)', fontsize=20, loc='top')
        plt.xlim([axis_inf[0], 1.0005])
        plt.ylim([axis_inf[1], 1.005])
        minorLocator = MultipleLocator(0.05)
        ax = plt.gca()
        ax.xaxis.set_minor_locator(minorLocator)
        if SPECIAL_DICT['LogScale'] in name:
            plt.yscale('log')
        else:
            ax.yaxis.set_minor_locator(minorLocator)


    plt.grid(which='both')
    hep.style.use('CMS')
    hep.cms.label('Preliminary')
    hep.cms.label(year='UL18')
    #plt.suptitle(name, horizontalalignment='center', verticalalignment='top', fontsize=25)
    plt.legend( loc='upper left') #labelcolor='linecolor',

    plt.savefig(f'{out_dir}/{name}.png', dpi = 200, bbox_inches='tight')
    if args.save:
        with open(f'{out_dir}/{name}.pickle', 'wb') as f:
            pickle.dump(fig_handle, f)
    if not args.not_show:
        plt.show()
    plt.close()

def load_dict(complete_dict, in_dict):
    """ Load the dictionary from the yaml file
    :param    complete_dict : string with the name of the file containing the paths to the models
    :param    in_dict : string with the name of the file containing the names of the models to load
    :return   info_dict : dictionary with the models to load
    """
    with open(complete_dict, 'r') as stream:
        loaded_dict=yaml.safe_load(stream)
    with open(in_dict, 'r') as stream:
        in_names=yaml.safe_load(stream)['networks']
    info_dict = defaultdict(list)
    for k, v in loaded_dict.items():
        if v[0] not in in_names:
            continue
        # dictionary with the path, the name of the model, the color and the line style
        info_dict[k].append(manager.dict())
        info_dict[k].append(v[0])
        info_dict[k].append(v[1])
        info_dict[k].append(v[2])
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
                if (SPECIAL_DICT['LabelsEpoch'] in filename)]
    best_files = [filename for filename in os.listdir(dir_name)
                if (SPECIAL_DICT['LabelsBest'] in filename)]

    # epochs to load
    if args.epochs == '-1':
            epoch_list=[len([k for k in files if SPECIAL_DICT['Primary'] in k])-1]
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
        if (args.only_primary and SPECIAL_DICT['Primary'] not in label_type) or label_type not in infile:
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
        if (args.only_primary and SPECIAL_DICT['Primary'] not in label_type) or label_type not in infile:
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
                EPOCHS_DICT[net_type][epoch][roc_type][info[1]]=(fpr, tpr, roc_auc, info[2], info[3])

                # load the mask for the labels (if present)
                try:
                    y_mask = file[PF_EXTRA_FTS[roc_type][0]].astype(bool)
                    if PF_EXTRA_FTS[roc_type][0] == SPECIAL_DICT['SpecialMask']:
                        y_mask = np.expand_dims(np.argmax(y_mask, axis=1), axis=1) == 0
                    y_true = y_true[y_mask[:, 0]]
                    y_score = y_score[y_mask[:, 0]]
                    # compute roc curve for each epoch
                    fpr, tpr, roc_auc=get_rates(y_true,y_score,
                                                labels[0], labels[1])
                    # save roc curve for each epoch
                    if history: info[0][f'{roc_type}_mask'][epoch]=(fpr, tpr, roc_auc)
                    EPOCHS_DICT[net_type][epoch][f'{roc_type}_mask'][info[1]]=(fpr, tpr, roc_auc, info[2], info[3])
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
        plt.plot(tpr,fpr,label=f'{roc_type} {info[1]} epoch #{epoch} (AUC=%0.4f)'% roc_auc)
    plt_fts(out_dir, f'ROC_{roc_type}_{info[1]}_{args.in_dict}_{net_type}_history', fig_handle)

def plotting_function(out_dir, epoch, roc_type, networks_1, name1, net_type, networks_2 = None, name2 = '', network_name='', line_style=None, line_color=None):
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

    if SPECIAL_DICT['Scatter_True-Reco'] not in roc_type:
        fig_handle = plt.figure(figsize=(20, 15))
        # loop over networks
        if isinstance(networks_1, mp.managers.DictProxy):
            for network, rates in networks_1.items():
                plt.plot(rates[1],rates[0],color=rates[3], linestyle=rates[4],label=f'{network} {epoch} (AUC=%0.4f)'% rates[2])
        elif isinstance(networks_1, tuple):
            rates=networks_1
            plt.plot(rates[1],rates[0],color=rates[3], linestyle=rates[4],label=f'{network_name} {name1} {epoch} (AUC=%0.4f)'% rates[2])

        if isinstance(networks_2, mp.managers.DictProxy):
            for network, rates in networks_2.items():
                plt.plot(rates[1],rates[0],color=rates[3] if line_color is None else line_color,
                        linestyle=rates[4] if line_style is None else line_style,
                        label=f'{network}(AUC=%0.4f)'% rates[2])
        elif isinstance(networks_2, tuple):
            rates=networks_2
            plt.plot(rates[1],rates[0],color=rates[3] if line_color is None else line_color,
                    linestyle=rates[4] if line_style is None else line_style,
                    label=f'{network_name} {name2} (AUC=%0.4f)'% rates[2])

        plt_fts(out_dir, f"ROC_{roc_type}_{args.in_dict}_{net_type}_{network_name}{epoch}", fig_handle, AXIS_INF[roc_type.replace('_mask', '').split('_')[-1]])

    else:
        out_dir_scatter = os.path.join(out_dir, 'Scatter')
        os.makedirs(out_dir_scatter, exist_ok=True)

        out_dir_true_reco = os.path.join(out_dir, 'True-Reco')
        os.makedirs(out_dir_true_reco, exist_ok=True)

        # loop over different types of features
        for i, limits in AXIS_LIMITS.items():
            # loop over networks
            for network, rates in networks_1.items():
                if i >= len(rates[0][0]):
                    continue
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
                            f'Scatter_{roc_type}{mask_name}_{limits[2]}_{network}_{args.in_dict}_{net_type}_{epoch}',
                            fig_handle)

            # loop over networks
            for network, rates in networks_1.items():
                if i >= len(rates[0][0]):
                    continue
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
                            f'True-Reco_{roc_type}{mask_name}_{limits[2]}_{network}_{args.in_dict}_{net_type}_{epoch}',
                            fig_handle)
        networks_1.clear()


def _main(net_type, out_dir, label_dict):
    """ Main function
    :param    net_type : string with the type of the network
    :param    out_dir : string with the name of the output directory
    :param    label_dict : dictionary with the labels
    """

    # create output directory
    out_dir = os.path.join(out_dir, f'{args.name}_{args.in_dict}_{net_type}_roc')
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
                if roc_type not in info[0].keys() or SPECIAL_DICT['Scatter_True-Reco'] == roc_type:
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
                            args=(out_dir, epoch, roc_type, networks_dict,
                                  get_middle_substring(roc_type.replace('_mask', '')), net_type))
            p.start()
            parallel_list.append(p)

        matching_groups = find_matching_suffix_groups(epoch_dict)
        for suffix, group in matching_groups.items():
            prefix = group[0].split('_')[0]

            mask = '_mask' if 'mask' in group[0] else ''
            extra_dict = {}
            extra_style = [('dotted', 'y'), ('dashed', 'y'), ('dashdot', 'y'), ('solid', 'y')]
            for i in range(1, len(group)):
                extra_dict[get_middle_substring(group[i].replace('_mask', ''))] = \
                list(epoch_dict[group[i]].values())[0][:3] + (extra_style[i-1][0],)+ (extra_style[i-1][1],)
            p=mp.Process(target=plotting_function,
                                args=(out_dir, epoch, f'{prefix}_comparison_{suffix}{mask}',
                                        epoch_dict[group[0]],
                                        get_middle_substring(group[0].replace('_mask', '')),
                                        net_type, extra_dict))
            p.start()
            parallel_list.append(p)

    # Join parallel
    for parallel_elem in parallel_list:
        parallel_elem.join()


if __name__ == '__main__':
    start=time.time()

    date_time = time.strftime('%Y%m%d-%H%M%S')
    main_out_dir = os.path.join(f'{args.out_path}roc_curve', f'{date_time}_{args.name}_{args.in_dict}_{args.type}_roc')
    print(f'Output directory: {main_out_dir}')

    parallel_list=[]
    for net_type in NET_TYPES:
        #label_dict=load_dict(f'{args.in_dict}_{net_type}.yaml')
        label_dict=load_dict(f'config/{args.complete_dict}.yaml', f'config/{args.in_dict}_{net_type}.yaml')
        p=mp.Process(target=_main,
                        args=(net_type, main_out_dir, label_dict))
        p.start()
        parallel_list.append(p)
    # Join parallel
    for parallel_elem in parallel_list:
        parallel_elem.join()

    print('starting comparison')
    parallel_list=[]

    if len(NET_TYPES) > 1:
        #label_dict=load_dict(f'{args.in_dict}_{NET_TYPES[0]}.yaml')
        label_dict=load_dict(f'config/{args.complete_dict}.yaml',f'config/{args.in_dict}_{NET_TYPES[0]}.yaml')
        input_name = list(label_dict.keys())[0]
        _, _, epoch_list, _ = create_lists(input_name)
        epoch_list.append('best')
        os.makedirs(f'{main_out_dir}/{args.name}_{args.in_dict}_{args.type}_roc', exist_ok=True)


        for info in label_dict.values():
            network=info[1].split('_')[0]
            for label_type, labels_info in ROC_TYPE_DICT.items():
                for roc_type in labels_info.keys():
                    if SPECIAL_DICT['Scatter_True-Reco'] == roc_type:
                        continue
                    for epoch in epoch_list:
                        for mask in ['', '_mask']:
                            #print(roc_type)
                            try:
                                for net_type in NET_TYPES:
                                    for key in EPOCHS_DICT[net_type][epoch][f'{roc_type}{mask}'].keys():
                                        EPOCHS_DICT[net_type][epoch][f'{roc_type}{mask}'][key.split('_')[0]] = \
                                        EPOCHS_DICT[net_type][epoch][f'{roc_type}{mask}'].pop(key)

                                p=mp.Process(target=plotting_function,
                                                    args=(f'{main_out_dir}/{args.in_dict}_{args.type}',
                                                    epoch, f'{roc_type}{mask}',
                                                    EPOCHS_DICT[NET_TYPES[0]][epoch][f'{roc_type}{mask}'][network],
                                                    NET_TYPES[0],
                                                    args.type, EPOCHS_DICT[NET_TYPES[1]][epoch][f'{roc_type}{mask}'][network],
                                                    NET_TYPES[1], network))
                                p.start()
                                parallel_list.append(p)
                            except KeyError:
                                pass
        # Join parallel
        for parallel_elem in parallel_list:
            parallel_elem.join()

    print('Total time:   ', time.time()-start)
