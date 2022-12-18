import numpy as np
import sklearn.metrics as _m
import matplotlib.pyplot as plt

infile_dict = {
    'CMSAK4/training/CMSAK4/PNXT_ef/20221218-131853_CMSAK4_PNXT_ef_ranger_lr0.01_batch2_sdfsdf/roc/y_bVSuds.npy': [[],'pnxt_ef'],
    'CMSAK4/training/CMSAK4/PN/20221218-132423_CMSAK4_PN_ranger_lr0.01_batch2_sdfsdf/roc/y_bVSuds.npy': [[],'pn'],
}

num_epoch=-1

for infile, info in infile_dict.items():
    with open(infile, 'rb') as f:
        try:
            while True:
                info[0].append(np.load(f))
        except EOFError and ValueError:
            pass


    fpr, tpr, thresh=_m.roc_curve(info[0][num_epoch][0], info[0][num_epoch][1])
    auc = _m.roc_auc_score(info[0][num_epoch][0], info[0][num_epoch][1])
    plt.plot(fpr,tpr,label=f'b vs uds {info[1]}, auc=%0.2f'% auc)


plt.plot([0, 1], [0, 1], 'k--', label='chance level (AUC = 0.5)')
plt.axis('square')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(f'b VS uds for epoch #{num_epoch}')
plt.legend()
plt.show()
plt.savefig(f'roc_curve_bVSuds_#{num_epoch}.png')
#print(epoch_list)
