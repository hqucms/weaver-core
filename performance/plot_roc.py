import numpy as np
import sklearn.metrics as _m
import matplotlib.pyplot as plt
import os
import sys
import re
import pickle
import argparse

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
    #    [[],[],'pnxt_ef_noweights_10e6_batch'],
    #'performance_20230121-000830_CMSAK4_PNXT_ranger_lr0.01_batch512_10e6_noweights_230k_attn':
    #    [[],[],'pnxt_noweights_10e6'],
    #'performance_20230123-102909_CMSAK4_PNXT_ef_ranger_lr0.01_batch512_10e6_noweights_230k_avg':
    #    [[],[],'pnxt_ef_avg_noweights_10e6', 'g-'],

    #'performance_20230111-124803_CMSAK4_PNXT_ef_ranger_lr0.01_batch512_10e6_noweights_230k_attn':
    #    [[],[],'pnxt_ef_attn_noweights_10e6', 'r-'],
    #'performance_20230119-235723_CMSAK4_PNXT_ranger_lr0.01_batch3072_10e6_noweights_230k_attn_batch3000':
    #    [[],[],'pnxt_noweights_10e6_batch', 'b-'],
    #'performance_20230122-094627_CMSAK4_PN_ranger_lr0.01_batch512_10e6_noweights_230k':
    #    [[],[],'pn_noweights_10e6', 'k-'],
    'performance_20230127-171444_CMSAK4_PNXT_ef_ranger_lr0.01_batch512_10e6_noweights_230k_attn_auxpf':
         [[],[],'pnxt_ef_attn_noweights_10e6_auxpf', 'g-'],
    #'../../CMSAK4/training/CMSAK4/PNXT_ef_aux/20230218-142254_CMSAK4_PNXT_ef_ranger_lr0.01_batch2_/performance_20230218-142254_CMSAK4_PNXT_ef_ranger_lr0.01_batch2_':
    #    [[],[],'pnxt_ef_aux', 'g-'],
}

epoch_list= [0,4,12,16,19]

roc_type_dict={
    "bVSuds":[[0,1], [4]],
    "bVSg":[[0,1], [5]],
    "bVSudsg":[[0,1], [4,5]],

}

def get_labels(y_true, y_score, labels_s, labels_b):
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
    y_true, y_score=get_labels(y_t,  y_s, l_s, l_b)
    fpr, tpr, thresh=_m.roc_curve(y_true, y_score)
    roc_auc = _m.roc_auc_score(y_true, y_score)
    return fpr, tpr, roc_auc

def plt_fts(roc_type, network, fig_handle):
    #plt.plot([0, 1], [0, 1], 'k--', label='chance level (AUC = 0.5)')
    #plt.axis('square')
    plt.xlabel('Efficency for b-jet (TP)')
    plt.ylabel('Mistagging prob (FP)')
    plt.ylim([0.0005, 1.05])
    plt.xlim([0.55, 1.0005])
    plt.yscale('log')
    plt.title(f'{roc_type}_{network}')
    plt.legend()
    if args.s:
        with open(f'roc_curve/roc_curve_{roc_type}_{network}.pickle', 'wb') as f:
            pickle.dump(fig_handle, f)
        plt.savefig(f'roc_curve/roc_curve_{roc_type}_{network}.png')
    if args.p:
        plt.show()
    plt.close()


if __name__ == "__main__":

    best_dict={}
    for roc_type in roc_type_dict.keys():
        best_dict[roc_type]={}

    for dir_name, info in label_dict.items():
        dir_name=os.path.join("input", dir_name)
        files = [filename for filename in os.listdir(dir_name)
                 if 'labels_epoch' in filename and ('bin' or 'regr' or 'clas') not in filename]
        print(files)
        best_file = [filename for filename in os.listdir(dir_name) if 'best' in filename and ('bin' or 'regr' or 'clas') not in filename][0]
        files.sort(key=lambda s: int(re.findall(r'\d+', s)[-2]))
        if args.e:
            for i, infile in enumerate(files):
                if i in epoch_list:
                    with open(os.path.join(dir_name,infile), 'rb') as f:
                        info[0].append(np.load(f)['y_true'])
                        info[1].append(np.load(f)['y_score'])

            print(info)
        for roc_type, labels in roc_type_dict.items():
            with open(os.path.join(dir_name,best_file), 'rb') as f:
                y_true_best=np.load(f)['y_true']
                y_score_best=np.load(f)['y_score']
                fpr, tpr, roc_auc=get_rates(y_true_best,y_score_best,
                                            labels[0], labels[1])
                best_dict[roc_type][info[2]]=(fpr, tpr, roc_auc, info[3])

            if args.e:
                fig_handle = plt.figure()
                for epoch_num in range(len(info[0])):
                    fpr, tpr, roc_auc=get_rates(
                        info[0][epoch_num],info[1][epoch_num],
                        labels[0], labels[1])

                    plt.plot(tpr,fpr,label=f'ROC {roc_type} {info[2]} epoch #{epoch_list[epoch_num]}, auc=%0.3f'% roc_auc)

                plt_fts(roc_type, info[2], fig_handle)

    for roc_type, net_dict in best_dict.items():
        fig_handle = plt.figure()
        for network, rates in net_dict.items():
            plt.plot(rates[1],rates[0],rates[3], label=f'ROC {network} best, auc=%0.4f'% rates[2])
        plt_fts(roc_type, "best_10e6", fig_handle)