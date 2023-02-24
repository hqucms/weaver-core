import matplotlib.pyplot as plt
import sys
import os
import re
import pickle
from collections import defaultdict

'''orig_stdout = sys.stdout
f = open('history.txt', 'w')
sys.stdout = f'''

'''
infile_dict = {
    #'performance_20221224-113421_CMSAK4_PNXT_ef_ranger_lr0.01_batch512_10e6_230k_attn': [[],[],[],[],'pnxt_ef_attn_10e6'],
    #'performance_20230110-171531_CMSAK4_PNXT_ef_ranger_lr0.01_batch512_10e6_cutweights_230k': [[],[],[],[],'pnxt_ef_attn_10e6_cutweights'],
    #'performance_20230113-001057_CMSAK4_PNXT_ef_ranger_lr0.001_batch512_10e6_looseweights_230k_attn_lr/': [[],[],[],[],'pnxt_ef_attn_10e6_cutweights_lr'],
    #'performance_20221224-113450_CMSAK4_PN_ranger_lr0.01_batch512_10e6': [[],[],[],[],'pn_10e6'],
    #'performance_20221224-120008_CMSAK4_PNXT_ranger_lr0.01_batch512_10e6': [[],[],[],[],'pnxt_10e6'],
    #'performance_20230115-155759_CMSAK4_PNXT_ef_ranger_lr0.01_batch512_10e6_labelweights_230k_attn': [[],[],[],[],'pnxt_ef_attn_10e6_labelweights'],

    #'performance_20230111-124803_CMSAK4_PNXT_ef_ranger_lr0.01_batch512_10e6_noweights_230k_attn': [defaultdict(list),'pnxt_ef_attn_10e6_noweights', 'r'],
    #'performance_20230118-110210_CMSAK4_PNXT_ef_ranger_lr0.01_batch3072_10e6_noweights_230k_attn_batch3000': [[],[],[],[],'pnxt_ef_attn_10e6_noweights_batch'],
    #'performance_20230119-235723_CMSAK4_PNXT_ranger_lr0.01_batch3072_10e6_noweights_230k_attn_batch3000': [[],[],[],[],'pnxt_attn_10e6_noweights_batch'],
    #('20230121-000830_CMSAK4_PNXT_ranger_lr0.01_batch512_10e6_noweights_230k_attn_0-19',): [[],[],[],[],'pnxt_attn_10e6_noweights'],
    #'performance_20230122-094627_CMSAK4_PN_ranger_lr0.01_batch512_10e6_noweights_230k': [[],[],[],[],'pn_attn_10e6_noweights'],
    #'performance_20230123-102909_CMSAK4_PNXT_ef_ranger_lr0.01_batch512_10e6_noweights_230k_avg': [[],[],[],[],'pnxt_ef_avg_10e6_noweights', 'g'],
    #'performance_20230127-171444_CMSAK4_PNXT_ef_ranger_lr0.01_batch512_10e6_noweights_230k_attn_auxpf': [defaultdict(list),'pnxt_ef_attn_10e6_noweights_auxpf', 'b'],
    #'performance_20230216-173113_CMSAK4_PNXT_ef_ranger_lr0.01_batch512_6_noweights_230k_attn_newdata_vtxmse': [defaultdict(list),'pnxt_ef_attn_10e6_noweights_auxpf', 'b'],
    'performance_20230218-141654_CMSAK4_PNXT_ef_ranger_lr0.01_batch512_50M_noweights_230k': [defaultdict(list),'pnxt_ef_attn_10e6_noweights_auxpf', 'b'],
}'''

infile_dict = {
'performance_20230218-141609_CMSAK4_PNXT_ef_ranger_lr0.01_batch512_50M_noweights_230k': [defaultdict(list),'ef', 'b'],
'performance_20230218-141615_CMSAK4_PNXT_ranger_lr0.01_batch512_50M_noweights_230k_selection' : [defaultdict(list),'pnxt', 'b'],
#'performance_20230218-141635_CMSAK4_PNXT_ef_ranger_lr0.01_batch512_50M_noweights_230k_regr' : [defaultdict(list),'regr', 'b'],
'performance_20230218-141654_CMSAK4_PNXT_ef_ranger_lr0.01_batch512_50M_noweights_230k' : [defaultdict(list),'clas', 'b'],
#'performance_20230218-141654_CMSAK4_PNXT_ef_ranger_lr0.01_batch512_50M_noweights_230k_noaux' : [defaultdict(list),'clas_noaux', 'b'],
#'performance_20230218-143000_CMSAK4_PNXT_ef_ranger_lr0.01_batch512_50M_noweights_230k_aux' : [defaultdict(list),'aux', 'b'],
#'performance_20230220-232814_CMSAK4_PN_ranger_lr0.01_batch512_50M_noweights_230k': [defaultdict(list),'pn', 'b'],
#'performance_20230221-201930_CMSAK4_PNXT_ef_ranger_lr0.01_batch512_50M_noweights_230k_bin': [defaultdict(list),'bin', 'b'],
#'performance_20230221-202602_CMSAK4_PNXT_ef_ranger_lr0.01_batch512_50M_noweights_230k_aux_tot': [defaultdict(list),'aux_tot', 'b'],
}

def plot(name):
    plt.xlabel('Epoch')
    plt.legend(loc=2, prop={'size': 15})
    plt.savefig(f'{name}.png')
    plt.show()

fig_handle = plt.figure()
for input_name, info in infile_dict.items():
    if isinstance(input_name, str):
        dir_name=os.path.join("input", input_name)
        infiles = [os.path.join(dir_name,filename) for filename in os.listdir(dir_name) if '.log' in filename]
        #print(infiles)
        infiles.sort(key=lambda s: int(re.findall(r'\d+', s)[-1]))
    elif isinstance(input_name, tuple):
        infiles=[os.path.join("input", "logs", f"{k}.log") for k in input_name]
        infiles.sort(key=lambda s: int(re.findall(r'\d+', s)[-1]))
    print(infiles)
    for infile in infiles:
        with open(infile) as f:
            f = f.readlines()

        for line in f:
            if 'Train AvgLoss' in line:
                info[0]["loss"].append(float(line.split('AvgLoss: ',1)[1].split(',')[0]))
                info[0]["acc"].append(float(line.split('AvgAcc: ',1)[1].split('\n')[0]))
                try:
                    info[0]["comb loss"].append(float(line.split('AvgCombLoss: ',1)[1].split(',')[0]))
                except IndexError:
                    pass
            elif 'Train AvgAuxLoss' in line:
                info[0]["aux loss"].append(float(line.split('AvgAuxLoss: ',1)[1].split(',')[0]))
                info[0]["aux acc pf"].append(float(line.split('AvgAuxAccPF: ',1)[1].split(',')[0]))
                try:
                    info[0]["aux dist"].append(float(line.split('AvgAuxDist: ',1)[1].split(',')[0]))
                    info[0]["aux acc pair"].append(float(line.split('AvgAuxAccPair: ',1)[1].split('\n')[0]))
                except IndexError:
                    pass
            elif 'validation metric' in line :
                info[0]["val metric"].append(float(line.split('validation metric: ',1)[1].split(' (')[0]))
                try:
                    info[0]["val loss"].append(float(line.split('validation loss: ',1)[1].split(' (')[0]))
                    info[0]["val comb loss"].append(float(line.split('validation combined loss: ',1)[1].split(' (')[0]))
                except IndexError:
                    pass
            elif 'validation aux metric' in line :
                info[0]["val aux metric pf"].append(float(line.split('validation aux metric PF: ',1)[1].split(' (')[0]))
                info[0]["val aux dist"].append(float(line.split('validation aux distance: ',1)[1].split(' (')[0]))
                try:
                    info[0]["val aux metric pair"].append(float(line.split('validation aux metric pair: ',1)[1].split(' (')[0]))
                except IndexError:
                    pass
                info[0]["val aux loss"].append(float(line.split('validation aux loss: ',1)[1].split(' (')[0]))


    '''print(f'loss {info[1]}:    ', info[0])
    print(f'accuracy {info[1]}:    ', info[1])
    print(f'validation metric {info[1]}:    ', info[2])
    print(f'validation loss {info[1]}:    ', info[3])'''


for _, info in infile_dict.items():
    for name, val in info[0].items():
        if 'aux' not in name and 'val' not in name:
            plt.plot(val, label=f'{name} {info[1]}')
plot('history')


for _, info in infile_dict.items():
    for name, val in info[0].items():
        if 'aux' in name and 'val' not in name:
            plt.plot(val, label=f'{name} {info[1]}')
plot('history_aux')

for _, info in infile_dict.items():
    for name, val in info[0].items():
        if 'aux' not in name and 'val' in name:
            plt.plot(val, label=f'{name} {info[1]}')
plot('history_val')

for _, info in infile_dict.items():
    for name, val in info[0].items():
        if 'aux' in name and 'val' in name:
            plt.plot(val, label=f'{name} {info[1]}')
plot('history_aux_val')





#with open('history.pickle', 'wb') as f:
#    pickle.dump(fig_handle, f)
'''
import pickle
figx = pickle.load(open('FigureObject.fig.pickle', 'rb'))

figx.show()
'''