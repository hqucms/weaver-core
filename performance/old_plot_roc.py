import numpy as np
import sklearn.metrics as _m
import matplotlib.pyplot as plt
import os
import re
import pickle

#np.set_printoptions(threshold=np.inf)

roc_dict = {
            "bVSuds":{
                #('PNXTef_y_bVSuds',): [[],'pnxt_ef_1'],
                #('y_bVSuds_avg',): [[],'pnxt_ef_avg'],
                #('CMSAK4_PN_20221219-221913_CMSAK4_PN_ranger_lr0.01_batch512_1e6_roc_y_bVSuds',): [[],'pn'],
                #('CMSAK4_PNXT_ef_20221219-205837_CMSAK4_PNXT_ef_ranger_lr0.01_batch512_1e6_attn_roc_y_bVSuds',): [[],'pnxt_ef_attn'],
                #('CMSAK4_PNXT_20221219-230206_CMSAK4_PNXT_ranger_lr0.01_batch512_1e6_roc_y_bVSuds',): [[],'pnxt'],
                #'../CMSAK4/training/CMSAK4/PNXT_ef/20221224-102506_CMSAK4_PNXT_ef_ranger_lr0.01_batch2_sdfsdf/performance_20221224-102506_CMSAK4_PNXT_ef_ranger_lr0.01_batch2_sdfsdf': [[],'pnxt_ef_avg'],
                'performance_20221220-181703_CMSAK4_PNXT_ef_ranger_lr0.01_batch512_1e6_230k_avg': [[],'pnxt_ef_avg_1e6'],
                'performance_20221220-193538_CMSAK4_PN_ranger_lr0.01_batch512_1e6': [[],'pn_1e6'],
                'performance_20221220-205122_CMSAK4_PNXT_ranger_lr0.01_batch512_1e6_230k': [[],'pnxt_1e6'],
                'performance_20221221-223336_CMSAK4_PNXT_ef_ranger_lr0.01_batch512_1e6_230k_attn': [[],'pnxt_ef_attn_1e6'],

            },
            "bVSg":{
                #('CMSAK4_PNXT_ef_20221219-205837_CMSAK4_PNXT_ef_ranger_lr0.01_batch512_1e6_attn_roc_y_bVSg',): [[],'pnxt_ef_attn'],
                #('CMSAK4_PN_20221219-221913_CMSAK4_PN_ranger_lr0.01_batch512_1e6_roc_y_bVSg',): [[],'pn'],
                #('CMSAK4_PNXT_20221219-230206_CMSAK4_PNXT_ranger_lr0.01_batch512_1e6_roc_y_bVSg',): [[],'pnxt'],
                'performance_20221220-181703_CMSAK4_PNXT_ef_ranger_lr0.01_batch512_1e6_230k_avg': [[],'pnxt_ef_avg_1e6'],
                'performance_20221220-193538_CMSAK4_PN_ranger_lr0.01_batch512_1e6': [[],'pn_1e6'],
                'performance_20221220-205122_CMSAK4_PNXT_ranger_lr0.01_batch512_1e6_230k': [[],'pnxt_1e6'],
                'performance_20221221-223336_CMSAK4_PNXT_ef_ranger_lr0.01_batch512_1e6_230k_attn': [[],'pnxt_ef_attn_1e6'],
            }
}



def plt_fts(roc_type, type, fig_handle):
    #plt.plot([0, 1], [0, 1], 'k--', label='chance level (AUC = 0.5)')
    #plt.axis('square')
    plt.xlabel('Efficency for b-jet (TP)')
    plt.ylabel('Mistagging prob (FP)')
    plt.ylim([0.0005, 1.05])
    plt.yscale('log')
    plt.title(f'{roc_type}_{type}')
    plt.legend()
    with open(f'roc_curve/roc_curve_{roc_type}_{type}.pickle', 'wb') as f:
        pickle.dump(fig_handle, f)
    plt.savefig(f'roc_curve/roc_curve_{roc_type}_{type}.png')
    #plt.show()
    plt.close()


best_dict={}
epoch_list= [0,1,5,13,17,20]

for roc_type, infile_dict in roc_dict.items():
    for dir_name, info in infile_dict.items():
        files = [filename for filename in os.listdir(dir_name) if roc_type in filename]
        files.sort(key=lambda s: int(re.findall(r'\d+', s)[-2]))
        #print(files)
        for i, infile in enumerate(files):
            if i in epoch_list:
                with open(os.path.join(dir_name,infile), 'rb') as f:
                    info[0].append(np.load(f))

        fig_handle = plt.figure()

        for epoch_num in range(0,len(info[0])):
            fpr, tpr, thresh=_m.roc_curve(info[0][epoch_num][0], info[0][epoch_num][1])
            roc_auc = _m.roc_auc_score(info[0][epoch_num][0], info[0][epoch_num][1])

            if epoch_num==0:
                best_dict[info[1]]=(fpr, tpr, roc_auc)
                continue

            plt.plot(tpr,fpr,label=f'ROC {roc_type} {info[1]} epoch #{epoch_list[epoch_num]-1}, auc=%0.3f'% roc_auc)

        plt_fts(roc_type, info[1], fig_handle)

    #print(best_dict)
    fig_handle = plt.figure()
    for net, rates in best_dict.items():
        plt.plot(rates[1],rates[0],label=f'ROC {roc_type} {net} best, auc=%0.3f'% rates[2])
    plt_fts(roc_type, "best_1e6", fig_handle)
