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

hep.style.use("CMS")

'''orig_stdout = sys.stdout
f = open('roc.txt', 'w')
sys.stdout = f'''

#np.set_printoptions(threshold=np.inf)

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=str, default='',
                    help='roc for various epochs')
parser.add_argument('--show', action='store_false', default=True,
                    help='show plots')
parser.add_argument('--save', action='store_true', default=False,
                    help='save plots')
parser.add_argument('--only-primary', action='store_true', default=False,
                    help='only compute the primary ROC')
args = parser.parse_args()

if args.epochs:
    epoch_list=[int(i) for i in args.epochs.split(',')]
else:
    epoch_list =[]

best_dict=defaultdict(defaultdict)

roc_type_dict=OrderedDict([
    ("pair_bin",{
        "pair_bin" : [[0], None]
    }),
    ("pf_clas",{
        "pf_clas" : [[0,2], [1,3]]
    }),
    ("pf_regr",{
        "pf_regr" : [None, None]
    }),
    #label
    ("primary",{
        "bVSuds":[[0,1], [4]],
        "bVSg":[[0,1], [5]],
        "bVSudsg":[[0,1], [4,5]]
    }),
])

axis_limits ={
    0: ((600, 600),(0,6,0,6), '_dist_pv'),
    1: ((300, 300),(-0.3,0.3,-0.3,0.3), '_vtx_x'),
    2: ((300, 300),(-0.3,0.3,-0.3,0.3), '_vtx_y'),
    3: ((600, 600),(-4,4,-4,4), '_vtx_z')
}

def get_labels(y_true, y_score, labels_s, labels_b):
    if labels_b is None:
        y_true_tot = y_true
        y_score_tot = y_score > 0.5
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

def get_rates(y_t, y_s, l_s, l_b):
    if l_s is None and l_b is None:
        fpr, tpr, roc_auc = y_s, y_t, np.nan
    else:
        y_true, y_score = get_labels(y_t,  y_s, l_s, l_b)
        fpr, tpr, thresh = _m.roc_curve(y_true, y_score)
        roc_auc = _m.roc_auc_score(y_true, y_score)
    return fpr, tpr, roc_auc

def plt_fts(roc_type, network, fig_handle, axis_lim=None, name=''):
    if 'regr' in roc_type:
        #plt.axis('square')
        plt.xlabel('error')
        if "error" not in network:
            plt.xlim([axis_lim[0], axis_lim[1]])
            plt.xlabel('True')
            plt.ylabel('Reco')
            plt.plot([-40, 40], [-40, 40], 'y--', label='True = Reco')
            plt.ylim([axis_lim[2], axis_lim[3]])
    else:
        plt.xlabel('Efficency for b-jet (TP)')
        plt.ylabel('Mistagging prob (FP)')
        plt.ylim([0.0005, 1.05])
        plt.xlim([0.55, 1.0005])
        #plt.yscale('log')

    hep.cms.label(rlabel="")
    plt.legend(labelcolor='linecolor')
    hep.cms.lumitext(f'ROC_{roc_type}_{network}{name}')
    #plt.title(f'{roc_type}_{network}{name}')
    plt.savefig(f'roc_curve/roc_curve_{roc_type}_{network}{name}.png', bbox_inches='tight')
    if args.save:
        with open(f'roc_curve/roc_curve_{roc_type}_{network}{name}.pickle', 'wb') as f:
            pickle.dump(fig_handle, f)
    if args.show:
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

if __name__ == "__main__":
    label_dict=load_dict('performance_comparison.yaml')

    for input_name, info in label_dict.items():
        dir_name=os.path.join("input", input_name)
        files = [filename for filename in os.listdir(dir_name)
                 if ('labels_epoch' in filename)]
        best_files = [filename for filename in os.listdir(dir_name)
                    if ('labels_best' in filename)]
        files.sort(key=lambda s: int(re.findall(r'\d+', s)[-1]))
        best_files.sort(key=lambda s: int(re.findall(r'\d+', s)[-1]))
        for infile in files:
            epoch = int(infile.split(".npz")[0][-2:] if infile.split(".npz")[0][-2].isnumeric() else infile.split(".npz")[0][-1])
            if epoch in epoch_list:
                for label_type, labels_info in roc_type_dict.items():
                    if args.only_primary and label_type != 'primary':
                        continue
                    if label_type in infile:
                        for roc_type, labels in labels_info.items():
                            with open(os.path.join(dir_name,infile), 'rb') as f:
                                info[0][label_type].append(np.load(f)['y_true'])
                                info[1][label_type].append(np.load(f)['y_score'])
                                break

        for best_file in best_files:
            for label_type, labels_info in roc_type_dict.items():
                if args.only_primary and label_type != 'primary':
                    continue
                if label_type in best_file:
                    with open(os.path.join(dir_name,best_file), 'rb') as f:
                        y_true_best=np.load(f)['y_true']
                        y_score_best=np.load(f)['y_score']
                        for roc_type, labels in labels_info.items():
                            #print(best_file, labels)
                            fpr, tpr, roc_auc=get_rates(y_true_best,y_score_best,
                                                        labels[0], labels[1])
                            best_dict[roc_type][info[2]]=(fpr, tpr, roc_auc, info[3])

                    break
        #print(best_dict)
        #print(info)
        for label_type, labels_info in roc_type_dict.items():
            if "regr" in roc_type:
                continue
            if len(info[0][label_type]) !=0:
                for roc_type, labels in labels_info.items():
                    if len(epoch_list) > 0:
                        fig_handle = plt.figure()
                        for num in range(len(info[0][label_type])):
                            fpr, tpr, roc_auc=get_rates(
                                info[0][label_type][num],info[1][label_type][num],
                                labels[0], labels[1])

                            plt.plot(tpr,fpr,label=f'ROC {roc_type} {info[2]} epoch #{epoch_list[num]}, auc=%0.3f'% roc_auc)

                        plt_fts(roc_type, input_name, fig_handle)

    for roc_type, net_dict in best_dict.items():
        if "regr" in roc_type:
            continue
        fig_handle = plt.figure()
        for network, rates in net_dict.items():
            plt.plot(rates[1],rates[0],rates[3],label=f'ROC {network} best, auc=%0.4f'% rates[2])
        plt_fts(roc_type, "best", fig_handle)

for roc_type, net_dict in best_dict.items():
    for i in range(4):
        if "regr" not in roc_type:
            continue
        fig_handle = plt.figure()
        for network, rates in net_dict.items():
            x_t=rates[1][:, i]
            y_r=rates[0][:, i]

            mask= (x_t!=0.)
            #x_t, y_r = x_t[mask], y_r[mask]

            plt.hist2d(x_t, y_r,  bins=axis_limits[i][0], cmap=plt.cm.jet)
            plt.colorbar().set_label('Density')
            plt_fts(roc_type, f"best_scatter_{network}", fig_handle, axis_limits[i][1], axis_limits[i][2])

        fig_handle = plt.figure()
        for network, rates in net_dict.items():
            x_t=rates[1][:, i]
            y_r=rates[0][:, i]

            mask= (x_t!=0.)
            #x_t, y_r = x_t[mask], y_r[mask]

            plt.hist((x_t-y_r), color= rates[3], bins=axis_limits[i][0][0], label=network, range=(-axis_limits[i][1][1], axis_limits[i][1][1]))
            plt_fts(roc_type, f"best_error_{network}", fig_handle, None, axis_limits[i][2])