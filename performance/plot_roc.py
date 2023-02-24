import numpy as np
import sklearn.metrics as _m
import matplotlib.pyplot as plt
import os
import sys
import re
import pickle
import argparse
from collections import defaultdict, OrderedDict

'''orig_stdout = sys.stdout
f = open('roc.txt', 'w')
sys.stdout = f'''

#np.set_printoptions(threshold=np.inf)

parser = argparse.ArgumentParser()
parser.add_argument('-e', action='store_true', default=False,
                    help='roc for various epochs')
parser.add_argument('-p', action='store_false', default=True,
                    help='print plots')
parser.add_argument('-s', action='store_false', default=True,
                    help='save plots')
args = parser.parse_args()

label_dict={
    #'performance_20230118-110210_CMSAK4_PNXT_ef_ranger_lr0.01_batch3072_10e6_noweights_230k_attn_batch3000':
    #    [defaultdict(list),defaultdict(list),'pnxt_ef_noweights_10e6_batch'],
    #'performance_20230121-000830_CMSAK4_PNXT_ranger_lr0.01_batch512_10e6_noweights_230k_attn':
    #    [defaultdict(list),defaultdict(list),'pnxt_noweights_10e6'],
    #'performance_20230123-102909_CMSAK4_PNXT_ef_ranger_lr0.01_batch512_10e6_noweights_230k_avg':
    #    [defaultdict(list),defaultdict(list),'pnxt_ef_avg_noweights_10e6', 'g-'],

    # 'performance_20230111-124803_CMSAK4_PNXT_ef_ranger_lr0.01_batch512_10e6_noweights_230k_attn':
    #    [defaultdict(list),defaultdict(list),'ef', 'r-'],
    # 'performance_20230119-235723_CMSAK4_PNXT_ranger_lr0.01_batch3072_10e6_noweights_230k_attn_batch3000':
    #    [defaultdict(list),defaultdict(list),'pnxt', 'b-'],
    # 'performance_20230122-094627_CMSAK4_PN_ranger_lr0.01_batch512_10e6_noweights_230k':
    #    [defaultdict(list),defaultdict(list),'pn', 'k-'],
    # 'performance_20230127-171444_CMSAK4_PNXT_ef_ranger_lr0.01_batch512_10e6_noweights_230k_attn_auxpf':
    #     [defaultdict(list),defaultdict(list),'clas', 'g-'],

    'performance_20230218-141615_CMSAK4_PNXT_ranger_lr0.01_batch512_50M_noweights_230k_selection':
        [defaultdict(list),defaultdict(list),'pnxt', 'k-'],
    'performance_20230218-141654_CMSAK4_PNXT_ef_ranger_lr0.01_batch512_50M_noweights_230k':
         [defaultdict(list),defaultdict(list),'clas', 'r'],
    # 'performance_20230218-141654_CMSAK4_PNXT_ef_ranger_lr0.01_batch512_50M_noweights_230k_noaux':
    #      [defaultdict(list),defaultdict(list),'clas_noaux', 'b'],
     'performance_20230218-141609_CMSAK4_PNXT_ef_ranger_lr0.01_batch512_50M_noweights_230k_selection':
        [defaultdict(list),defaultdict(list),'ef', 'g-'],
    # 'performance_20230218-143000_CMSAK4_PNXT_ef_ranger_lr0.01_batch512_50M_noweights_230k_aux':
    #      [defaultdict(list),defaultdict(list),'aux', 'r'],
    # 'performance_20230218-141635_CMSAK4_PNXT_ef_ranger_lr0.01_batch512_50M_noweights_230k_regr_selection':
    #     [defaultdict(list),defaultdict(list),'regr', 'c'],
        # 'performance_20230221-201930_CMSAK4_PNXT_ef_ranger_lr0.01_batch512_50M_noweights_230k_bin':
        # [defaultdict(list),defaultdict(list),'bin', 'c'],

}

epoch_list= [0,1,2,3,4,12,16,19]

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
    }),7
    #primary
    ("label",{
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
            plt.plot([-40, 40], [-40, 40], 'y--', label='True == Reco')
            plt.ylim([axis_lim[2], axis_lim[3]])
    else:
        plt.xlabel('Efficency for b-jet (TP)')
        plt.ylabel('Mistagging prob (FP)')
        plt.ylim([0.0005, 1.05])
        plt.xlim([0.55, 1.0005])
        plt.yscale('log')
    plt.legend()
    plt.title(f'{roc_type}_{network}{name}')
    if args.s:
        with open(f'roc_curve/roc_curve_{roc_type}_{network}{name}.pickle', 'wb') as f:
            pickle.dump(fig_handle, f)
        plt.savefig(f'roc_curve/roc_curve_{roc_type}_{network}{name}.png')
    if args.p:
        plt.show()
    plt.close()



if __name__ == "__main__":
    #print(roc_type_dict)
    for input_name, info in label_dict.items():
        dir_name=os.path.join("input", input_name)
        files = [filename for filename in os.listdir(dir_name)
                 if ('labels_epoch' in filename)]
        best_files = [filename for filename in os.listdir(dir_name)
                    if ('labels_best' in filename)]
        files.sort(key=lambda s: int(re.findall(r'\d+', s)[-1]))
        best_files.sort(key=lambda s: int(re.findall(r'\d+', s)[-1]))
        if args.e:
            for infile in files:
                epoch = int(infile.split(".npz")[0][-2:] if infile.split(".npz")[0][-2].isnumeric() else infile.split(".npz")[0][-1])
                #print(epoch)
                if epoch in epoch_list:
                    for label_type, labels_info in roc_type_dict.items():
                        if label_type in infile:
                            for roc_type, labels in labels_info.items():
                                with open(os.path.join(dir_name,infile), 'rb') as f:
                                    info[0][label_type].append(np.load(f)['y_true'])
                                    info[1][label_type].append(np.load(f)['y_score'])
                                    break

        for best_file in best_files:
            for label_type, labels_info in roc_type_dict.items():
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
        print(best_dict)
        for label_type, labels_info in roc_type_dict.items():
            if len(info[0][label_type]) !=0:
                for roc_type, labels in labels_info.items():
                    if args.e:
                        fig_handle = plt.figure()
                        for num in range(len(info[0][label_type])):
                            fpr, tpr, roc_auc=get_rates(
                                info[0][label_type][num],info[1][label_type][num],
                                labels[0], labels[1])

                            plt.plot(tpr,fpr,label=f'ROC {roc_type} {info[2]} epoch #{epoch_list[num]}, auc=%0.3f'% roc_auc)

                        plt_fts(roc_type, input_name, fig_handle)

    for roc_type, net_dict in best_dict.items():
        if "regr" not in roc_type:
            fig_handle = plt.figure()
            for network, rates in net_dict.items():
                plt.plot(rates[1],rates[0],rates[3]+'-',label=f'ROC {network} best, auc=%0.4f'% rates[2])
            plt_fts(roc_type, "best", fig_handle)

for roc_type, net_dict in best_dict.items():
    for i in range(4):
        if "regr" in roc_type:
            fig_handle = plt.figure()
            for network, rates in net_dict.items():
                x_t=rates[1][:, i]
                y_r=rates[0][:, i]
                mask= (x_t!=0.)
                x_t =x_t[mask]
                y_r=y_r[mask]
                #if i == 2 or i==1: print(x_t, y_r, type(x_t))
                #plt.plot(x, y,rates[3]+'.',label=f'ROC {network} best, auc=%0.4f'% rates[2])
                plt.hist2d(x_t, y_r,  bins=axis_limits[i][0], cmap=plt.cm.jet)
            plt.colorbar().set_label('Density')
            plt_fts(roc_type, "best_scatter", fig_handle, axis_limits[i][1], axis_limits[i][2])

            fig_handle = plt.figure()
            for network, rates in net_dict.items():
                x_t=rates[1][:, i]
                y_r=rates[0][:, i]
                mask= (x_t!=0.)
                x_t =x_t[mask]
                y_r=y_r[mask]

                #if i == 2 or i==1: #print(x_t, y_r, type(x_t))
                #plt.plot(x, y,rates[3]+'.',label=f'ROC {network} best, auc=%0.4f'% rates[2])

                plt.hist((x_t-y_r),  bins=axis_limits[i][0][0], label=network, range=(-axis_limits[i][1][1], axis_limits[i][1][1]))
                plt_fts(roc_type, f"best_error_{network}", fig_handle, None, axis_limits[i][2])